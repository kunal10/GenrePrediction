require 'rnn'
require 'cunn'
require 'cutorch'
require 'xlua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

cmd:option('-batch_size', 64, 'BatchSize')
cmd:option('-test_data', 'features/test_data.t7', 'test data file')
cmd:option('-test_targets', 'features/test_data_labels.t7', 'test data labels')
cmd:option('-model', 'models/blstm_weights2000.t7', 'model to be evaluated')
cmd:option('-output_file', 'results/predictionsw.txt', 'model predictions')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

batchSize = opt.batch_size
cutorch.setDevice(opt.gpuid)

-- Open file for writing outputs.
file = io.open(opt.output_file, "w")

local rnn = torch.load(opt.model)
-- Enable evaluate mode
rnn:evaluate()

-- As per comment here: https://github.com/hsheil/rnn-examples/blob/master/part2/main.lua this is essential
rnn:remember('both')

-- print(rnn)

-- build criterion
criterion = nn.ClassNLLCriterion()

-- Load inputs and targets
inputs = torch.load(opt.test_data)
targets = torch.load(opt.test_targets)

-- Convert to cuda compatible versions
if (opt.gpuid > 0) then
  criterion = criterion:cuda()
  rnn = rnn:cuda()
end

local function writeToFile(predicted, target)
    for i = 1,predicted:size(1) do
          file:write(string.format("%f\n", predicted[i][1]))
    end
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

-- Iterate over all test data
local testSize = #inputs - 1
local totalerr = 0
local totalaccuracy = 0
for i = 1,testSize do
    xlua.progress(i, testSize)
    
    local predicted = rnn:forward(inputs[i])
    writeToFile(predicted, targets[i])
    -- printDebugInfo(predicted, targets[i])
    
    local batcherr, batchaccuracy = computeBatchErrorAndAccuracy(predicted, targets[i])
    print(string.format("Batch: %d err = %f, accuracy = %f", i, batcherr, batchaccuracy))
    totalerr = totalerr + batcherr
    totalaccuracy = totalaccuracy + batchaccuracy
end

print(string.format("Avg Err = %f, Avg Accuracy = %f", totalerr/testSize, totalaccuracy/testSize))

-- Close the output file
print("Finised writing output. Closing the output file")
file:close()
