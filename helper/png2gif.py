
def make_gif(iter):
    import glob
    from PIL import Image, ImageChops
    import numpy
    from PIL.GifImagePlugin import getheader, getdata
    import gifmaker
    sequence = []
    # fileList = glob.glob('../hko_fast/1900it6850outputn_*')
    fileList = []
    path = '../hko_lstm_baseline_nopeep/'
    
    
    for i in range(15):
        name = path + 'trainImg/it' + str(iter) + 'outputn_' + str(i + 1) + '.png'
        fileList.append(name)
    print(fileList)

    for file in fileList:
        print(file)
        im = Image.open(file)
        img = numpy.asarray(im)
        print(img)
        sequence.append(im)
    name = path + str(iter) + "out.gif"
    fp = open(name, "wb")
    print('save as '+ name)
    gifmaker.makedelta(fp, sequence)
    fp.close()
    # gifmaker.save_gif(sequence, 'input.gif')

make_gif(2000)
