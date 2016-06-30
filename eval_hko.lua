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
log.level = opt.evalLogLevel or "info"
paths.dofile('data_hko.lua')
log.outfile = opt.testLogDir..'eval_hko_log'

log.info('[init] load model file: ', opt.modelFile)
log.info('[init] load model parameters: ', opt.modelPara)


dofile(opt.modelFile)
parameters, gradParameters = model:getParameters()
log.trace('[init] para of model init: ', parameters:sum())

local para = torch.load(opt.modelPara):cuda()
log.trace('[init] para of model load: ', para:sum())
parameters:fill(1)
parameters:cmul(para)
-- parameters = para:clone()

parameters, gradParameters = model:getParameters()
log.trace('[init] para of model after set: ', parameters:sum())

local paraEnc, _ = enc:getParameters()
local paraDec, _ = dec:getParameters()
log.trace('[init] para of enc + dec: ', paraEnc:sum(), paraDec:sum())

startTestUtils()
assert(opt.init)
opt.useGpu = true 
log.info('[init] useGpu: ', opt.useGpu)

local criterion = nn.SequencerCriterion(nn.MSECriterion())
log.info('[init] set criterion ', criterion)

if opt.useGpu then
    require 'cunn'
    criterion = criterion:cuda()
end
-- local model = torch.load(opt.modelEncDec)
local enc = model.modules[1]
local dec = model.modules[2]

enc = enc:double()
dec = dec:double()
if opt.useGpu then
    enc = enc:cuda()
    dec = dec:cuda()
end

enc:zeroGradParameters()
dec:zeroGradParameters()


log.info(string.format("[init] ==> evaluating model: enc: %s, dec: %s", opt.modelEnc, opt.modelDec))
datasetSeq = getdataSeqHko('test')

local scores = {}
scores.POD = torch.Tensor(opt.outputSeqLen):zero()
scores.FAR = torch.Tensor(opt.outputSeqLen):zero()
scores.CSI = torch.Tensor(opt.outputSeqLen):zero()
scores.correlation = torch.Tensor(opt.outputSeqLen):zero()
scores.rainRmse =  torch.Tensor(opt.outputSeqLen):zero()

local POD = 0
local FAR = 0
local CSI = 0
local correlation = 0
local rainRmse = 0

-- prepare zero input for decoder:
local inputDecTensor = torch.Tensor(opt.batchSize, opt.imageDepth, opt.imageHeight, opt.imageWidth):fill(0)
if opt.useGpu then
    inputDecTensor = inputDecTensor:cuda()
end
local inputDecTable = {}
for i = 1, opt.outputSeqLen do
    table.insert(inputDecTable, inputDecTensor)
end
local loss = 0
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
    forwardConnect(enc, dec)
    local output = dec:forward(inputDecTable)
    scores = SkillScore(scores, output, target, opt.threthold)

    POD = scores.POD[opt.outputSeqLen] / iter
    FAR = scores.FAR[opt.outputSeqLen] / iter
    CSI = scores.CSI[opt.outputSeqLen] / iter
    correlation = scores.correlation[opt.outputSeqLen] / iter
    rainRmse = scores.rainRmse[opt.outputSeqLen] / iter

    if(math.fmod(iter, opt.testSaveIter) == 1) then
        log.trace('[saveimg] input, output and target save to ', opt.imgDir)
        OutputToImage(inputEncTable, iter, 'input')
        OutputToImage(output, iter, 'output')
        OutputToImage(target, iter, 'target')
    end
    
    loss = loss + criterion:updateOutput(output, target) / opt.outputSeqLen

    if(math.fmod(iter, opt.scoreDisplayIter) == 1) then
        log.info(string.format('@loss %.4f, iter %d / %d ', loss / iter, iter, opt.maxTestIter))
        log.info(string.format("[score] POD %.3f \t FAR %.3f \t CSI %.3f \t correlation %.3f \t rainRmse %.3f", POD, FAR, CSI, correlation, rainRmse))
    end
end

allPOD = scores.POD:div(opt.maxTestIter)
allFAR = scores.FAR:div(opt.maxTestIter)
allCSI = scores.CSI:div(opt.maxTestIter)
allCorrelation = scores.correlation:div(opt.maxTestIter)
allRainRmse = scores.rainRmse:div(opt.maxTestIter)


log.info("POD: ", allPOD)
log.info("FAR: ", allFAR)
log.info("CSI; ", allCSI)
log.info("correlation: ", allCorrelation)
log.info("rainRmse: ", allCorrelation)
log.info("@done evaluate")
