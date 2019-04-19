import torch
from torch.autograd import Variable
import numpy as np
from FCN import FCNs
from FCN import VGGNet 
from TestData import testloader
import time
import visdom



if __name__ == "__main__":
# the input to the model is of shape N*3*160*160 tensor, N presents the number of samples
    index = 0
    start = time.time()
    model = torch.load('checkpoints/fcn_model_95.pt')
    model.cuda()
    vis = visdom.Visdom()
    # model = model.cpu()
    with torch.no_grad():  
        print(len(testloader)) 
        for item in testloader:
            index += 1
            input = item['A']
            y = item['B']
            input = input.cuda()
            y = y.cuda()
            output = model(input)
            print(type(input))
            print(input.size())
            #TODO why he is using sigmoid?
            # output = torch.sigmoid(output)


            if np.mod(index, 20) ==1:
                output_np = output.cpu().data.numpy().copy()
                output_np = np.argmin(output_np, axis=1)
                y_np = y.cpu().data.numpy().copy()
                y_np = np.argmin(y_np, axis=1)
                # vis.close()
                vis.images(output_np[:, None, :, :], opts=dict(title='pred')) 
                vis.images(y_np[:, None, :, :], opts=dict(title='label')) 


    end = time.time()
    execution_time = end - start
    # TODO global variable for batch size 
    test_set_size = 16*len(testloader)
    print("test set size  :", test_set_size)
    print("execution time :", execution_time)


    # y is the output, of shape N*2*160*160, 2 present the class, [1 0] for background [0 1] for handbag




