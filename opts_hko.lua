local log = loadfile("../helper/log.lua/log.lua")()

opt = {}
opt.onMac = true

if opt.onMac then
	opt.gpuflag = false
else
	opt.gpuflag = true
end

-- general options:
opt.dir     = 'of_record' -- subdirectory to save experiments in
opt.seed    = 1250         -- initial random seed
opt.imageDepth = 1

-- Model parameters:
opt.inputSizeW = 50   -- width of each input patch or image
opt.inputSizeH = 50  -- width of each input patch or image
opt.nSeq      = 20
opt.batchSize = 5
opt.inputSeqLen = 5
opt.outputSeqLen = 3
opt.trainSamples = 2037 * 4
opt.transf = 2       -- number of parameters for transformation; 6 for affine or 3 for 2D transformation

opt.nFilters  = {4, 32}      --9,45} -- number of filters in the encoding/decoding layers
opt.nFiltersMemory   = {1, 17} --{45,60}

opt.maskStride = opt.imageH/opt.inputSizeH -- assume h = w

opt.kernelSize       = 3 -- size of kernels in encoder/decoder layers
opt.kernelSizeMemory = 3

--------------------------- training confirguration
opt.paraInit = 0.01
opt.parametersInitStd = 0.01
opt.trainIter = torch.floor(opt.trainSamples / opt.batchSize) 
opt.validIter = torch.floor(opt.validSamples / opt.batchSize) 
opt.selectStep = 2


--------------------------
opt.kernelSizeFlow   = 15
opt.padding   = torch.floor(opt.kernelSize / 2) -- pad input before convolutions
opt.dmin = -0.5 -- -0.5
opt.dmax = 0.5 -- 0.5
opt.stride = 1 --opt.kernelSizeMemory -- no overlap
opt.constrWeight = {0, 1, 0.001}

opt.memorySizeW = opt.inputSizeW
opt.memorySizeH = opt.inputSizeH

opt.save         = true -- save models

if not paths.dirp(opt.dir) then
   os.execute('mkdir -p ' .. opt.dir)
end

opt.nSamples = 2037 * 4 -- number of frames in total
opt.nSeq = 20 -- length of the sequence in one trainSeq, input + output
opt.imageHeight = 100 -- number of rows and cols in one image matrix
opt.imageWidth = 100 
opt.imageDepth = 1
opt.dataFile = "../helper/trainData.txt"
opt.dataPath = "../data/hko/"
