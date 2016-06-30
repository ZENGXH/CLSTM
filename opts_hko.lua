
local log = loadfile("log.lua")()

opt = {}
opt.onMac = true
opt.init = false -- call startTrainUtils or startTestUtils for init
-- general options:
opt.useGpu = true
opt.useSeqLSTM = false 
opt.backend = 'cudnn'
opt.seed = 1

-- data config
opt.imageHeight = 100 -- number of rows and cols in one image matrix
opt.imageWidth = 100 
opt.imageDepth = 1
opt.listFileTrain = "all_data_list_train.txt"
opt.listFileTest = "all_data_list_test.txt"
opt.dataPath = "/home/xhzeng/project/myTorch/"

opt.inputSeqLen = 5
opt.outputSeqLen = 15
opt.totalSeqLen = opt.inputSeqLen + opt.outputSeqLen
opt.encoderRho = 1
opt.decoderRho = opt.outputSeqLen
opt.dataOverlap = false
-- number of parameters for transformation; 6 for affine or 3 for 2D transformation

-- model confirguration
opt.inputSizeW = 100   -- width of each input patch or image
opt.inputSizeH = 100  -- width of each input patch or image
opt.nFiltersMemory = {1, 17, 45}  --{45,60}

opt.memorySizeW = opt.inputSizeW
opt.memorySizeH = opt.inputSizeH
opt.batchSize = 3 

opt.kernelSize  = 3 -- size of kernels in encoder/decoder layers
opt.padding  = torch.floor(opt.kernelSize / 2) -- pad input before convolutions
opt.stride = 1

-- training confirguration
opt.modelFile = 'model_hko_2toconv.lua'
opt.trainLogLevel = "info"
opt.saveIter = 20 -- save img
opt.displayIter = 100 -- show loss
opt.modelSaveIter = 2000
opt.maxIter = 40000
opt.gradClip = 50
opt.lr = 1e-3
opt.saveDir = './' -- '/mnt/ficusengland/ssd1/xhzeng2/hko_lstm_baseline_nopeep/'
opt.saveDirTrainImg = opt.saveDir..'trainImg/'
opt.save = true -- save models

opt.modelDir = opt.saveDir.."model/"

-- for test 
opt.modelPara = opt.modelDir.."encDecpara_iter_6001.bin"
opt.contIter = 6002
opt.test = true
opt.evalLogLevel = "trace"
opt.modelTestID = 7000
opt.modelEnc = opt.modelDir..'enc_iter_'..tostring(opt.modelTestID)..'.bin'
opt.modelDec = opt.modelDir..'dec_iter_'..tostring(opt.modelTestID)..'.bin'
opt.maxTestIter = 1980
opt.testLogDir = opt.saveDir..'test_log'
opt.saveDirTestImg = opt.saveDir..'test_img/' 
opt.testSaveIter = 10

opt.scoreDisplayIter = 5 -- show loss
-- confirguration

-- for different log level
opt.dataLoaderLogLevel = "info"

-- optical flow part
opt.kernelSizeFlow = 3
opt.transf = 2
opt.memorySizeW = opt.inputSizeW
opt.memorySizeH = opt.inputSizeH
opt.dmin = - 0.5
opt.dmax = 0.5

if not paths.dirp(opt.saveDir) then
    os.execute('mkdir -p ' .. opt.saveDir)
end
if not paths.dirp(opt.modelDir) then
    os.execute('mkdir -p ' .. opt.modelDir)
end
