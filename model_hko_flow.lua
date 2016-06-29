require 'rnn'
require 'ConvLSTM'
require 'utils'
require 'flow'
-- require 'stn'
require 'BilinearSamplerBHWD'
require 'DenseTransformer2D' -- AffineGridGeneratorOpticalFlow2D
require 'cunn'

local log = loadfile("log.lua")()

-- decoder convolution layer
enc_conv = nn.SpatialConvolution(opt.nFiltersMemory[1], opt.nFiltersMemory[2], 3, 3, 1, 1, 1, 1)

-- memory lstm
encoder_0 = nn.ConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2], opt.decoderRho, 3, 3, opt.stride, opt.batchSize)

-- model start with decoder convolution layer
model = nn.Sequential():add(enc_conv)

-- keep last frame to apply optical flow on
local branch_base = nn.Sequential()
-- transpose feature map for the sampler 
-- branch_base:add(nn.Transpose({1,3},{1,2})) 
-- this case default input after encoder conv is (1, 100, 100), but then in our case it is B D H W -> H D B W -< D H B W
-- it should be B D H W -> B D W H -> B H W D
branch_base:add(nn.Transpose({3, 4}, {2, 4}))

local branch_flow = nn.Sequential()
branch_flow:add(encoder_0)
branch_flow:add(flow)
local concat = nn.ConcatTable()
concat:add(branch_base):add(branch_flow)
model:add(concat)

-- add sampler
model:add(nn.BilinearSamplerBHWD())
-- B H W D -> B D W H -> B D H W

--model:add(nn.Transpose({1,3},{2,3})) -- untranspose the result!!
model:add(nn.Transpose({2, 4},{3, 4})) -- untranspose the result!!

dec_conv = nn.SpatialConvolution(opt.nFiltersMemory[2], opt.nFiltersMemory[1], 3, 3, 1, 1, 1, 1)
model:add(dec_conv)

-- <GLOBAL>: model, branch_memory, opticalFlow
-- add(nn.Sequencer(decoderConv))
branch_memory = nn.Sequential():add(enc_conv):add(encoder_0)

if opt.useGpu then
    require 'cunn'
    model = model:cuda()
    branch_memory = branch_memory:cuda()
    log.info('[model init] using gpu')
end
log.trace(model)
log.trace(branch_memory)

log:trace("[init] load model and branch_memory done@")

