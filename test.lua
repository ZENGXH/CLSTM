require 'nn'
require 'cunn'
require 'cutorch'
require 'rnn'
require 'ConvLSTM'
require 'utils'
local log = loadfile("log.lua")()
dofile 'opts_hko.lua'
g = 1024 * 1024 * 1024

function test1()
    collectgarbage()
    local usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace(usemem/g - totalmem/g)
    local a = torch.Tensor(1, 1, 100, 100)
    local input = a:cuda()

    usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace(usemem/g - totalmem/g)

    local encoder_0 = nn.ConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2], opt.encoderRho, 3, 3, opt.stride, opt.batchSize)
    encoder_0 = encoder_0:cuda()

    usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace('init', usemem/g - totalmem/g)
    local output_enc = encoder_0:forward(input)

    usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace("enc forward: ", usemem/g - totalmem/g)
    collectgarbage()
end



function test2()
    collectgarbage()
    local usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace(usemem/g  - totalmem/g)
    local a = torch.Tensor(1, 1, 100, 100)
    local input = a:cuda()

    usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace(usemem/g - totalmem/g)

    local encoder_0 = nn.ConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2], opt.encoderRho, 3, 3, opt.stride, opt.batchSize)
    encoder_0 = encoder_0:cuda()

    local encoder_0_seq = nn.Sequencer(encoder_0):cuda()
    usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace('init', usemem/g)

    local output_enc = encoder_0_seq:forward({input})

    usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace("enc forward: ", usemem/g  - totalmem/g)
    collectgarbage()
end

function test3()
    collectgarbage()
    local usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace(usemem/g - totalmem/g)
    local a = torch.Tensor(1, opt.nFiltersMemory[2], 100, 100)
    local input = a:cuda()

    usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace(usemem/g - totalmem/g)

    local encoder_0 = nn.ConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2], opt.encoderRho, 3, 3, opt.stride, opt.batchSize)
    encoder_0 = encoder_0:cuda()

    local encoder_0_seq = nn.Sequencer(encoder_0):cuda()
    usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace('init', usemem/g - totalmem/g)

    local output_enc = encoder_0_seq:forward({input})

    usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace("enc forward: ", usemem/g - totalmem/g)
    collectgarbage()
end

function testSpatial()
    collectgarbage()
    usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace(totalmem/g - usemem/g)
    a = torch.Tensor(1, opt.nFiltersMemory[2], 100, 100)
    input = a:cuda()

    usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace(totalmem/g - usemem/g)

    encoder_0 = nn.SpatialConvolution(opt.nFiltersMemory[2], opt.nFiltersMemory[2], 3, 3, 1, 1, 1, 1)
    -- encoder_0 = nn.ConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2], opt.encoderRho, 3, 3, opt.stride, opt.batchSize)
    encoder_0 = encoder_0:cuda()

    usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace('init', totalmem/g - usemem/g)

    output_enc = encoder_0:forward(input)

    usemem, totalmem = cutorch.getMemoryUsage(1)
    log.trace("enc forward: ", totalmem/g - usemem/g )
    collectgarbage()
end

testSpatial()
--[[
output_dec = dec:forward(input)

local usemem, totalmem = cutorch.getMemoryUsage(1)
log.trace("dec forward: ", usemem/g)

------------------
print('output_enc', output_enc)
print('output_dec', output_dec)
enc:backward(output_enc, input)

local usemem, totalmem = cutorch.getMemoryUsage(1)
log.trace("enc backward: ", usemem/g)

dec:backward(output_dec, input)

local usemem, totalmem = cutorch.getMemoryUsage(1)
log.trace("dec backward: ", usemem/g)
]]--
