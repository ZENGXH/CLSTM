
local log = loadfile("log.lua")()

opt = {}
opt.onMac = true

-- general options:
opt.imageDepth = 1
opt.useGpu = true
opt.useSeqLSTM = false 

-- Model parameters:
opt.inputSizeW = 100   -- width of each input patch or image
opt.inputSizeH = 100  -- width of each input patch or image
opt.nFiltersMemory = {1, 17}  --{45,60}

opt.memorySizeW = opt.inputSizeW
opt.memorySizeH = opt.inputSizeH
opt.batchSize = 2

opt.inputSeqLen = 5
opt.outputSeqLen = 15
opt.totalSeqLen = opt.inputSeqLen + opt.outputSeqLen
opt.encoderRho = 1
opt.decoderRho = opt.outputSeqLen

-- number of parameters for transformation; 6 for affine or 3 for 2D transformation
-- model confirguration
opt.kernelSize  = 3 -- size of kernels in encoder/decoder layers
opt.padding  = torch.floor(opt.kernelSize / 2) -- pad input before convolutions
opt.stride = 1

-- training confirguration
opt.maxIter = 100000
opt.lr = 0.001
opt.saveDir = '/mnt/ficusengland/ssd1/xhzeng2/hko_lstm_baseline/'
opt.save = true -- save models
opt.nSamples = 2037 * 4 -- number of frames in total
opt.nSeq = 20 -- length of the sequence in one trainSeq, input + output
opt.imageHeight = 100 -- number of rows and cols in one image matrix
opt.imageWidth = 100 
opt.imageDepth = 1
opt.listFileTrain = "all_data_list_train.txt"
opt.listFileTest = "all_data_list_test.txt"

opt.dataPath = "../"
opt.modelDir = opt.saveDir.."model/"
if not paths.dirp(opt.saveDir) then
    os.execute('mkdir -p ' .. opt.saveDir)
end
if not paths.dirp(opt.modelDir) then
    os.execute('mkdir -p ' .. opt.modelDir)
end

-- for test 
opt.modelEnc = opt.modelDir..'enc_iter_9000.bin'
opt.modelDec = opt.modelDir..'dec_iter_9000.bin'
opt.maxTestIter = 2000
opt.testLogDir = opt.saveDir..'test/'
if not paths.dirp(opt.testLogDir) then
    os.execute('mkdir -p ' .. opt.testLogDir)
end

-- confirguration
opt.seed = 1
