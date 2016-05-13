require 'rnn'
require 'cunn'
require 'cutorch'
require 'xlua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-hidden_size', 1000, 'Size of LSTM unit output')
cmd:option('-feature_size', 300, 'Size of word2vec input features to LSTM')
cmd:option('-batch_size', 64, 'batch_size')
cmd:option('-num_classes', 10, 'number of genres')
cmd:option('-num_iterations', 10000, 'number of training iterations')

cmd:option('-train_data', 'features/train_data.t7', 'training data file')
cmd:option('-train_targets', 'features/train_data_labels.t7', 'training data labels')
cmd:option('-model_prefix', 'models/lstm_weights', 'model prefix file')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)

-- hyper-parameters 

-- Number of steps to backpropogate gradients.
featureSize = opt.feature_size -- Length of feature vector
hiddenSize = opt.hidden_size
batchSize = opt.batch_size
numClasses = opt.num_classes
lr = 0.01

-- Load inputs and targets
inputs = torch.load(opt.train_data)
targets = torch.load(opt.train_targets)
assert(#inputs == #targets)
numBatches = #inputs 
numTrainBatches = numBatches
-- numTrainBatches = torch.ceil(0.9 * numBatches)
numIterations = opt.num_iterations

print(#inputs)
print(#targets)

-- forward rnn
local fwd = nn.FastLSTM(featureSize, hiddenSize)

local frnn = nn.Sequencer(fwd)

local rnn = nn.Sequential()
   :add(nn.SplitTable(1,3))
   :add(frnn)
   :add(nn.SelectTable(-1))
   :add(nn.Linear(hiddenSize, numClasses))
   :add(nn.LogSoftMax())

--according to http://arxiv.org/abs/1409.2329 this should help model performance 
rnn:getParameters():uniform(-0.1, 0.1)

---- Tip as per https://github.com/Element-Research/rnn/issues/125
rnn:zeroGradParameters()

-- As per comment here: https://github.com/hsheil/rnn-examples/blob/master/part2/main.lua this is essential
rnn:remember('both')

-- build criterion
weights = torch.Tensor({0.1, 0.02, 0.07, 0.028, 0.023, 0.029, 0.02, 0.1, 0.05, 0.017})
criterion = nn.ClassNLLCriterion(weights)
-- criterion = nn.ClassNLLCriterion()

-- Convert to cuda
if (opt.gpuid > 0) then
  criterion = criterion:cuda()
  rnn = rnn:cuda()
end

local function computeBatchErrorAndAccuracy(predicted, targets)
    local err = criterion:forward(predicted, targets)
    local max, indices = torch.max(predicted, 2)
    local matched_predictions = torch.sum(torch.eq(indices[{{},{1}}], targets))
    local accuracy = matched_predictions / batchSize;
    -- print(predicted)
    -- print(max)
    -- print(indices)
    -- print(targets)
    -- print(torch.eq(indices[{{},{1}}], targets))
    -- print(matched_predictions)
    return err, accuracy
end

print(rnn)
rnn:training()
-- Iterate over all input batches and learn params.
for i = 1,numIterations do
    --xlua.progress(i, numIterations)
    local index = (i % numTrainBatches)
    
    if (index == 0) then 
      index = numTrainBatches
    end
    local outputs = rnn:forward(inputs[index])
    -- printDebugInfo(outputs, targets[i])
    
    --local err = criterion:forward(outputs, targets[index])
    -- print(string.format("Iteration %d ; err = %f ", i, err))

    -- 3. backward sequence through rnn (i.e. backprop through time)
    local gradOutputs = criterion:backward(outputs, targets[index])
    local gradInputs = rnn:backward(inputs[index], gradOutputs)

    -- 4. update
    rnn:updateParameters(lr)
    rnn:zeroGradParameters()
    rnn:forget()

    -- Evaluate training error and accuracy for this iteration.
    local batcherr, batchaccuracy = computeBatchErrorAndAccuracy(outputs, targets[index])
    print(string.format("Iteration %d ; Val err = %f Val accuracy = %f", i, batcherr, batchaccuracy))

    if (i % 1000 == 0) then
      lr = lr * 0.8
    end

    if (i % 2000 == 0) then
      torch.save(opt.model_prefix .. i .. '.t7', rnn)
    end
end

print('Saving Trained Model')
-- torch.save(opt.model_prefix .. numIterations .. '.t7', rnn)
