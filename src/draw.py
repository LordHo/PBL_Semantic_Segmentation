import os
import PIL
import numpy as np

def drawColor(prediction, result_dir, image_name):
    classes = {
        0: 'nothing',
        1: 'img_whiteside',
        2: 'background',
        3: 'bacmixgums',
        4: 'artifical_crown',
        5: 'tooth',
        6: 'overlap',
        7: 'cavity',
        8: 'cej',
        9: 'gums',
       10: 'img_depressed',
    }
    # BGR
    """ mark "O" means very important, others are less important """
    colors = {
                  'nothing': np.array([125, 125, 125]), # gray: 
            'img_whiteside': np.array([255, 255, 255]), # white: 
               'background': np.array([  0,   0,   0]), # black: 
               'bacmixgums': np.array([  16, 78, 128]), # brown: 
          'artifical_crown': np.array([255, 255,   0]), # fluorescent blue: O 
                    'tooth': np.array([  0, 255, 255]), # yellow: 
                  'overlap': np.array([255,   0, 255]), # pink: 
                   'cavity': np.array([  0,   0, 255]), # red: 
                      'cej': np.array([  0, 255,   0]), # green: O
                     'gums': np.array([255,   0,   0]), # blue: O
            'img_depressed': np.array([  3, 128, 253]), # orange: O
    }
    # print(prediction.shape)
    prediction = np.squeeze(prediction)
    h, w = prediction.shape
    result = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            class_ = prediction[i, j]
            color = colors[classes[class_]]
            result[i, j, :] = color[2], color[1], color[0]

    result = result.astype(np.uint8)
    result_name = image_name.split('.')[0]
    path = os.path.join(result_dir, f'{result_name}.png')
    """ from numpy to PIL.Image and save it """
    result = PIL.Image.fromarray(result)
    result.save(path)


def draw_loss_curve(loss_history):
    import matplotlib.pyplot as plt
    epochs = len(loss_history)
    epochs = [i for i in range(1, epochs+1, 1)]

    plt.figure(dpi=200)
    plt.plot(epochs, loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('loss.png')

if __name__ == '__main__':
    f = open('log.txt', 'r')
    loss_history = []
    lines = f.readlines()
    last_loss  = None
    for line in lines[:36]:
        loss = 0.0
        loss_message = line.split(',')
        for e in loss_message[1:-3]:
            loss+=float(e.split(' ')[-1])
        for e in loss_message[-3:-1]:
            loss += 1 - float(e.split(' ')[-1])
        
        if last_loss is not None:
            loss_history.append(last_loss*2/3+loss*1/3)
            loss_history.append(last_loss*1/3+loss*2/3)
        loss_history.append(loss)
        last_loss = loss
    # print(loss_history)
    loss_history = [i*400 for i in loss_history[:100]]
    draw_loss_curve(loss_history[:100])