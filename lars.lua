require 'cutorch'

torch.manualSeed(0)
cutorch.manualSeed(0)

local X_mean = torch.CudaTensor()
local X_stdev = torch.CudaTensor()

local Y_mean = torch.CudaTensor()
local Y_stdev = torch.CudaTensor()

local c = torch.CudaTensor()
local cabs = torch.CudaTensor()
local csign = torch.CudaTensor()
local Ga = torch.CudaTensor()
local Ga_inv = torch.CudaTensor()
local ones = torch.CudaTensor()
local w = torch.CudaTensor()
local W = torch.CudaTensor()
local Wabs = torch.CudaTensor()
local u = torch.CudaTensor()
local a = torch.CudaTensor()
local total_dist = torch.CudaTensor()
local dist_positive_denominator = torch.CudaTensor()
local dist_negative_denominator = torch.CudaTensor()
local dist_mask = torch.CudaByteTensor()
local active_mask = torch.CudaByteTensor()
local Xa = torch.CudaTensor()

local bias_term = torch.CudaTensor()

-- "Closeness" values
local lambda = 1 - 1e-10
local epsilon = 1e-2

function lars(X, Y, total_weight, model_select, lasso_path)
  local samples = X:size(1)
  local inputs = X:size(2)
  local recovering = false

  -- Normalize X:
  X_mean:resize(inputs):mean(X, 1)
  X:csub(X_mean:view(1, inputs):expandAs(X))
  X_stdev:resize(inputs):std(X, 1)
  X:cdiv(X_stdev:view(1, inputs):expandAs(X))

  -- Normalize Y:
  Y_mean:resize(1):mean(Y, 1)
  Y:csub(Y_mean:resize(1, 1):expandAs(Y))

  -- Get covariances; this will have size (inputs) x 1
  c:resize(inputs, 1):zero()
  c:addmm(0, c, 1, X:t(), Y)

  cabs:resizeAs(c)
  csign:resizeAs(c)

  total_dist:resize(2 * inputs, 1)
  local dist_positive = total_dist:narrow(1, 1, inputs)
  local dist_negative = total_dist:narrow(1, inputs+1, inputs)

  dist_positive_denominator:resizeAs(dist_positive)
  dist_negative_denominator:resizeAs(dist_negative)

  active_mask:resize(inputs):zero()
  local active = {}

  local just_removed = false

  -- Weight matrix
  W:resize(inputs, 1):zero()
  Wabs:resizeAs(W):zero()

  cabs:copy(c):abs()

  local cmax = cabs:max()
  for i=1,inputs do
    if cabs[i][1] > cmax * lambda - epsilon then
      table.insert(active, i)
      active_mask[i] = 1
    end
  end

  while #active ~= inputs do
    csign:copy(c):sign()

    -- Select the active columns of the inputs matrix
    Xa:resize(samples, #active)
    for i=1,#active do
      Xa:narrow(2, i, 1):copy(
        X:narrow(2, active[i], 1)
      ):mul(csign[active[i]][1])
    end

    -- Compute the equiangular vector.
    --
    -- TODO this could be sped up by a linear factor with torch.gesv, unsure
    -- whether this would be faster or slower than CUDA inverse.
    Ga:resize(#active, #active):zero()
    Ga:addmm(0, Ga, 1, Xa:t(), Xa)

    Ga_inv:inverse(Ga)

    ones:resize(#active):fill(1)
    w:resize(#active):zero()
    w:addmv(0, w, 1, Ga_inv, ones)

    local A = 1 / math.sqrt(w:sum())
    u:resize(samples)
    u:addmv(0, u, A, Xa, w)

    a:resize(inputs)
    a:addmv(0, a, 1, X:t(), u)

    -- Compute distances to equal correlations
    dist_positive:copy(c):add(cmax)
    dist_positive_denominator:copy(a):add(A)
    dist_positive:cdiv(dist_positive_denominator)

    dist_negative:copy(c):csub(cmax)
    dist_negative_denominator:copy(a):csub(A)
    dist_negative:cdiv(dist_negative_denominator)

    local gamma = math.huge
    local new_index = -1
    for i=1,inputs do
      if recovering or active_mask[i] == 0 then
        if dist_positive[i][1] > 0 and dist_positive[i][1] < gamma then
          gamma = dist_positive[i][1]
          new_index = i
        end
        if dist_negative[i][1] > 0 and dist_negative[i][1] < gamma then
          gamma = dist_negative[i][1]
          new_index = i
        end
      end
    end

    recovering = false

    if new_index == -1 then
      print('Breaking due to some error.')
      break
    end

    -- Lasso path modification, if specified
    local subgamma = math.huge

    inactive_indices = {}
    if lasso_path then
      for i=1,#active do
        local weight = W[active[i]][1]
        local delta = csign[active[i]][1] * w[i] * A
        local candidate_subgamma = -weight / delta

        if candidate_subgamma > 0 and candidate_subgamma < subgamma then
          subgamma = candidate_subgamma
          inactive_indices[active[i]] = true
        elseif candidate_subgamma > 0 and candidate_subgamma < subgamma - epsilon then
          inactive_indices[active[i]] = true
        end
      end
    end

    if subgamma < gamma then
      gamma = subgamma
      recovering = true
    end

    -- Increase parameters by w * gamma
    for i=1,#active do
      W[active[i]]:add(w[i] * csign[active[i]][1] * A * gamma)
    end

    -- Decrease correlations by a * gamma
    c:csub(a:mul(gamma))
    cmax = cmax - A * gamma

    --[[
    if lasso_removal then
      active_mask[active[index_to_remove] ] = 0
      table.remove(active, index_to_remove)
      just_removed = true
    else
      -- Add new variable to the active set
      if active_mask[new_index] == 0 then
        table.insert(active, new_index)
        active_mask[new_index] = 1
      end
      just_removed = false
    end]]

    -- Recompute actives
    active = {}
    active_mask:zero()

    cabs:copy(c):abs()
    cmax = cabs:max()
    for i=1,inputs do
      if cabs[i][1] > cmax * lambda - epsilon then
        -- During lasso path computation, exclude things that are crossing zero
        if not inactive_indices[i] then
          table.insert(active, i)
          active_mask[i] = 1
        end
      end
    end

    print(#active)

    if Wabs:copy(W):abs():sum() > total_weight then
      break
    end
  end

  if model_select then
    return active
  else
    return W:cmul(X_stdev:view(inputs, 1):expandAs(W))
  end
end

-- TESTING

local X_true = torch.CudaTensor(1000, 50):normal()
local W_true = torch.CudaTensor(50, 1):zero()

for i=1,10 do
  W_true[i]:fill(5) --normal()
end

local Y_true = torch.CudaTensor(1000, 1):uniform()
Y_true:addmm(10, Y_true, 1, X_true, W_true)

local ols = torch.inverse(X_true:t() * X_true) * X_true:t() * Y_true
print('OLS:')
print(ols)

print('LARS, 50:')
print(lars(X_true, Y_true, 45, true, false))

print('LASSO, 50:')
print(lars(X_true, Y_true, 45, true, true))
