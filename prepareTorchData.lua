require 'nn'
require 'optim'
require 'torch'
require 'math'
require 'cunn'
require 'cutorch'
require 'xlua'
npy4th = require 'npy4th'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-train_labels_file','features/train_labels','path to train labels file')
cmd:option('-test_labels_file','features/test_labels','path to test labels file')
cmd:option('-train_features_file','features/train_features.npy','path to train features')
cmd:option('-test_features_file','features/test_features.npy','path to test features')
cmd:option('-batch_size', 64, 'batch_size')
cmd:option('-seq_len', 100, 'seq_len')

cmd:option('-out_train_data', 'features/train_data.t7', 'training data file')
cmd:option('-out_train_labels', 'features/train_data_labels.t7', 'training data labels')
cmd:option('-out_test_data', 'features/test_data.t7', 'test data file')
cmd:option('-out_test_labels', 'features/test_data_labels.t7', 'test data labels')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

cutorch.setDevice(opt.gpuid)

local train_labels_filename = opt.train_labels_file
local test_labels_filename = opt.test_labels_file
local train_features_filename = opt.train_features_file
local test_features_filename = opt.test_features_file
local batch_size = opt.batch_size
local seq_len = opt.seq_len

----------------------------------------------------------------------
-- 1. load numpy features
train_features = npy4th.loadnpy(train_features_filename)
print(torch.type(train_features))
test_features = npy4th.loadnpy(test_features_filename)
print(torch.type(test_features))
local num_train_egs = train_features:size(2)
local num_test_egs = test_features:size(2)
print('Num Training egs: ' .. num_train_egs)
print('Num Test egs: ' .. num_test_egs)

----------------------------------------------------------------------
-- 2. Load labels
print('Train Labels File: ' .. train_labels_filename)
local train_labels_file = io.open(train_labels_filename, 'r'); 
train_labels = torch.zeros(num_train_egs)
local num_train_songs = 0
local train_song_index = 1
for line in train_labels_file:lines() do
  if not line then
    break
  end
  local song_id, genre_tag, song_len, feature_len = unpack(line:split(" "));
  print(song_id, genre_tag, song_len, feature_len)
  print('Adding label for song id:' .. song_id)
  train_labels[train_song_index] = tonumber(genre_tag)
  train_song_index = train_song_index + 1
  num_train_songs = num_train_songs + 1
end
train_labels_file:close()

print('Test Labels File: ' .. test_labels_filename)
local test_labels_file = io.open(test_labels_filename, 'r'); 
test_labels = torch.zeros(num_test_egs)
local num_test_songs = 0
local test_song_index = 1
for line in test_labels_file:lines() do
  if not line then
    break
  end
  local song_id, genre_tag, song_len, feature_len = unpack(line:split(" "));
  print(song_id, genre_tag, song_len, feature_len)
  print('Adding label for song id:' .. song_id)
  test_labels[test_song_index] = tonumber(genre_tag)
  test_song_index = test_song_index + 1
  num_test_songs = num_test_songs + 1
end
test_labels_file:close()

-- assert(num_train_egs == num_train_songs)
-- assert(num_test_egs == num_test_songs)
print('Number of train songs: ' .. num_train_songs);
print('Number of train labels: ')
print(train_labels:size())
print('Number of test songs: ' .. num_test_songs);
print('Number of test labels: ')
print(test_labels:size())

---------------------------------------------------------------------

-- training data
local num_train_batches = torch.ceil(num_train_songs / batch_size)
local train_inputs, train_targets = {}, {}
for i = 1,num_train_batches do
  xlua.progress(i, num_train_batches)
  -- train_inputs[i], train_targets[i] = {}, {}
  print('processing train batch: ' .. i)
  local batch_start_index = (i-1) * batch_size + 1
  local batch_end_index = math.min(num_train_songs, i * batch_size)
  table.insert(train_inputs, train_features[{ {}, {batch_start_index, batch_end_index} , {} }]:clone():cuda())
  table.insert(train_targets, train_labels[{ {batch_start_index, batch_end_index} }]:clone():cuda())
  collectgarbage()
end
torch.save(opt.out_train_data, train_inputs)
torch.save(opt.out_train_labels, train_targets)

-- test data
local num_test_batches = torch.ceil(num_test_songs/ batch_size)
local test_inputs, test_targets = {}, {}
for i = 1,num_test_batches do
  xlua.progress(i, num_test_batches)
  -- test_inputs[i], test_targets[i] = {}, {}
  print('processing test batch: ' .. i)
  local batch_start_index = (i-1) * batch_size + 1
  local batch_end_index = math.min(num_test_songs, i * batch_size)
  table.insert(test_inputs, test_features[{ {}, {batch_start_index, batch_end_index} , {} }]:clone():cuda())
  table.insert(test_targets, test_labels[{ {batch_start_index, batch_end_index} }]:clone():cuda())
  collectgarbage()
end

torch.save(opt.out_test_data, test_inputs)
torch.save(opt.out_test_labels, test_targets)
print(torch.type(train_inputs[1]))
print(torch.type(train_targets[1]))
print(train_inputs[1]:size())
print(train_targets[1]:size())
