import torch
from torch.autograd import Variable
import numpy as np
from FCN import FCNs
from FCN import VGGNet 
from TestData import testloader
import time
import visdom
import cv2


if __name__ == "__main__":
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name

    model = torch.load('checkpoints/fcn_model_95.pt')
    model = model.cuda()
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    with torch.no_grad():  

        # cv2.namedWindow("output_frame1", 0)
        # cv2.resizeWindow("output_frame1", 640,480)
        # Read until video is completed
        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:

                cv2.imshow('Frame', frame)
                frame = cv2.resize(frame, (160, 160))
                frame_tensor = torch.from_numpy(frame)
                frame_tensor = frame_tensor.permute(2, 0, 1)
                input = torch.empty(1,3,160,160)
                input[0] = frame_tensor
                input = input.cuda()
                output = model(input)
                #TODO why he is using sigmoid?
                output = torch.sigmoid(output)

                output = output[0]
                output_np = np.argmin(output_np, axis=1)²
            
                output1 =  torch.empty(1, 160, 160)
                output2 =  torch.empty(1, 160, 160)
                output1[0] = output[0]
                output2[0] = output[1]

                output1 = output1.permute(1, 2, 0)
                output2 = output2.permute(1, 2, 0)
                print(output1.size())
                print(output2.size())

                output1 = output1.cpu()
                output2 = output2.cpu()

                output_frame1 = output1.numpy()
                output_frame2 = output2.numpy()

                # #Thresholding
                # ret,output_frame1 = cv2.threshold(output_frame1,(122.0/255.0),1,cv2.THRESH_BINARY)
                # print((127.0/255.0))
                # ret,output_frame2 = cv2.threshold(output_frame2,(122.0/255.0),1,cv2.THRESH_BINARY)

                # print("output :****************************************************************************************************************")
                # print(output_frame1)
                # for i in output_frame1:
                #     print(i)

                # output_frame1 = cv2.resize(output_frame1, (320, 320))
                # output_frame2 = cv2.resize(output_frame2, (320, 320))

            
                # # Display the resulting frame
                cv2.imshow('output_frame1',output_frame1)
                cv2.imshow('output_frame2',output_frame2)
            
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            
            # Break the loop
            else: 
                break
        
        # When everything done, release the video capture object
        cap.release()
        
        # Closes all the frames
        cv2.destroyAllWindows()










# # the input to the model is of shape N*3*160*160 tensor, N presents the number of samples
#     index = 0
#     start = time.time()
#     model = torch.load('checkpoints/fcn_model_95.pt')
#     model.cuda()
#     vis = visdom.Visdom()
#     # model = model.cpu()
#     with torch.no_grad():  
#         print(len(testloader)) 
#         for item in testloader:
#             index += 1
#             input = item['A']
#             y = item['B']
#             input = input.cuda()
#             y = y.cuda()
#             output = model(input)
#             #TODO why he is using sigmoid?
#             # output = torch.sigmoid(output)


#             if np.mod(index, 20) ==1:
#                 output_np = output.cpu().data.numpy().copy()
#                 output_np = np.argmin(output_np, axis=1)
#                 y_np = y.cpu().data.numpy().copy()
#                 y_np = np.argmin(y_np, axis=1)
#                 # vis.close()
#                 vis.images(output_np[:, None, :, :], opts=dict(title='pred')) 
#                 vis.images(y_np[:, None, :, :], opts=dict(title='label')) 


#     end = time.time()
#     execution_time = end - start
#     # TODO global variable for batch size 
#     test_set_size = 16*len(testloader)
#     print("test set size  :", test_set_size)
#     print("execution time :", execution_time)






