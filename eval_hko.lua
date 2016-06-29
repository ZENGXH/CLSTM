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

startTestUtils()
assert(opt.init)
opt.useGpu = false 
log.info('[init] useGpu: ', opt.useGpu)

local criterion = nn.SequencerCriterion(nn.MSECriterion())
log.info('[init] set criterion ', criterion)
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
    
    scores = SkillScore(scores, output, target, opt.threthold)
    -- POD = (POD + scores.POD) / 2
    -- FAR = (FAR + scores.FAR) / 2
    -- CSI = (CSI + scores.CSI) / 2
    -- correlation = (correlation + scores.correlation) / 2

    POD = scores.POD[opt.outputSeqLen] / iter
    FAR = scores.FAR[opt.outputSeqLen] / iter
    CSI = scores.CSI[opt.outputSeqLen] / iter
    correlation = scores.correlation[opt.outputSeqLen] / iter
    rainRmse = scores.rainRmse[opt.outputSeqLen] / iter
    if(math.fmod(iter, opt.testSaveIter) == 0) then
        log.trace('[saveimg] input, output and target save to %s')
        OutputToImage(inputEncTable, iter, 'input')
        OutputToImage(output, iter, 'output')
        OutputToImage(target, iter, 'target')
    end


    log.info(string.format("iter: %d POD %.3f \t FAR %.3f \t CSI %.3f \t correlation %.3f \t rainRmse %.3f", iter, POD, FAR, CSI, correlation, rainRmse))
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
