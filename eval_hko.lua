require 'nn'
require 'image'
require 'paths'
require 'rnn'
require 'torch'
require 'ConvLSTM'
require 'utils'
require 'cunn'
require 'cutorch'

local log = loadfile('log.lua')()
paths.dofile('opts_hko.lua')
paths.dofile('data_hko.lua')
log.outfile = opt.testLogDir..'eval_hko_log'

opt.useGpu = false

local criterion = nn.SequencerCriterion(nn.MSECriterion())
if opt.useGpu then
    require 'cunn'
    criterion = criterion:cuda()
end
enc = torch.load(opt.modelEnc)
dec = torch.load(opt.modelDec)
enc = enc:double()
dec = dec:double()
if opt.useGpu then
    enc = enc:cuda()
    dec = dec:cuda()
end

enc:zeroGradParameters()
dec:zeroGradParameters()
log.info('[init] ==> evaluating model')
datasetSeq = getdataSeqHko('test')
local POD = 0
local FAR = 0
local CSI = 0
local correlation = 0
local rainRmse = 0
for iter = 1, opt.maxTestIter do
    local sample = datasetSeq[iter]
    local data = sample[1]  
    if opt.useGpu then
        data = data:cuda()
    end

    local inputEncTable = {}
    for i = 1, opt.inputSeqLen do 
        table.insert(inputEncTable, data[{{}, {i}, {}, {}, {}}]:select(2,1))
    end

    local target = {}
    for i = 1, opt.outputSeqLen do
        table.insert(target, data[{{}, {opt.inputSeqLen + i}, {}, {}, {}}] )
    end

    -- in shape (15, 8, 1, 100, 100)
    local output_enc = enc:forward(inputEncTable)
    log.trace('[fw] forward encoder done@ output')
    forwardConnect(enc, dec)
    inputDecTensor = torch.Tensor(opt.batchSize, opt.imageDepth, opt.imageHeight, opt.imageWidth):fill(0)
    if opt.useGpu then
        inputDecTensor = inputDecTensor:cuda()
    end

    local inputDecTable = {}
    for i = 1, opt.outputSeqLen do
        table.insert(inputDecTable, inputDecTensor)
    end

    local output = dec:forward(inputDecTable)

    local loss = criterion:updateOutput(output, target)
    log.info('@loss ', loss, ' iter ', iter, ' / ', opt.maxTestIter)
    local scores = SkillScore(output[opt.outputSeqLen], target[opt.outputSeqLen], opt.threthold)
    -- POD = (POD + scores.POD) / 2
    -- FAR = (FAR + scores.FAR) / 2
    -- CSI = (CSI + scores.CSI) / 2
    -- correlation = (correlation + scores.correlation) / 2

    POD = scores.POD
    FAR = scores.FAR
    CSI = scores.CSI
    correlation = scores.correlation
    rainRmse = scores.rainRmse
    log.info('@POD ', POD, ' FAR ', FAR, ' CSI ', CSI, ' correlation ', correlation, 'rainRmse', rainRmse)
end

