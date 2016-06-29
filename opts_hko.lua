
local log = loadfile("log.lua")()

opt = {}
opt.onMac = true
opt.init = false -- call startTrainUtils or startTestUtils for init
-- general options:
opt.imageDepth = 1
opt.useGpu = true
opt.useSeqLSTM = false 
opt.backend = 'cudnn'
-- Model parameters:
opt.inputSizeW = 100   -- width of each input patch or image
opt.inputSizeH = 100  -- width of each input patch or image
opt.nFiltersMemory = {1, 32, 45}  --{45,60}

opt.memorySizeW = opt.inputSizeW
opt.memorySizeH = opt.inputSizeH
opt.batchSize = 1 

opt.inputSeqLen = 5
opt.outputSeqLen = 15
opt.totalSeqLen = opt.inputSeqLen + opt.outputSeqLen
opt.encoderRho = 1
opt.decoderRho = opt.outputSeqLen

-- number of parameters for transformation; 6 for affine or 3 for 2D transformation
-- model confirguration
opt.kernelSize  = 7 -- size of kernels in encoder/decoder layers
opt.padding  = torch.floor(opt.kernelSize / 2) -- pad input before convolutions
opt.stride = 1

opt.gradClip = 50
-- training confirguration
opt.trainLogLevel = "info"
opt.saveIter = 20
opt.maxIter = 40000
opt.lr = 1e-5
opt.saveDir = './' -- '/mnt/ficusengland/ssd1/xhzeng2/hko_lstm_baseline_nopeep/'
opt.saveDirTrainImg = opt.saveDir..'trainImg/'
opt.save = true -- save models
opt.nSamples = 2037 * 4 -- number of frames in total
opt.imageHeight = 100 -- number of rows and cols in one image matrix
opt.imageWidth = 100 
opt.imageDepth = 1
opt.listFileTrain = "all_data_list_train.txt"
opt.listFileTest = "all_data_list_test.txt"

opt.dataPath = "/home/xhzeng/project/myTorch/"
opt.modelDir = opt.saveDir.."model/"
if not paths.dirp(opt.saveDir) then
    os.execute('mkdir -p ' .. opt.saveDir)
end
if not paths.dirp(opt.modelDir) then
    os.execute('mkdir -p ' .. opt.modelDir)
end

-- for test 
opt.evalLogLevel = "trace"
opt.modelTestID = 7000
opt.modelEnc = opt.modelDir..'enc_iter_'..tostring(opt.modelTestID)..'.bin'
opt.modelDec = opt.modelDir..'dec_iter_'..tostring(opt.modelTestID)..'.bin'
opt.maxTestIter = 1980
opt.testLogDir = opt.saveDir..'test/'
opt.saveDirTestImg = opt.saveDir..'test_img/' 
opt.testSaveIter = 1

-- confirguration
opt.seed = 1

-- for different log level
opt.dataLoaderLogLevel = "info"

-- optical flow part
opt.kernelSizeFlow = 3
opt.transf = 2
opt.memorySizeW = opt.imageWidth
opt.memorySizeH = opt.imageHeight
opt.dmin = -0.5
opt.dmax = 0.5

