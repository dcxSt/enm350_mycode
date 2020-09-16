# since we only care about the field and the force exerted by these charges and not the force felt by these charges we won't bother with the mass...
# each charge will be represented by a length-3 list [charge, x, y], everything is in the plane for this question so we don't even need a z axis

import math
import numpy as np
import matplotlib.pyplot as plt

def polar_to_cartesian(theta,r=1.0):
    return [r*math.cos(theta),r*math.sin(theta)]

# takes n slots and m charges
def set_slots(n):
    theta = 2*math.pi / n
    slots = [polar_to_cartesian(i*theta) for i in range(n)]
    return slots

# takes slots, number of charges (m), and set up the charges as stipulated in the question
def set_charges(m,slots):
    n = len(slots)
    q = 1.0/m 
    charges = []
    for i in range(m):
        charges.append([q,slots[i][0],slots[i][1]])
    for i in range(m,n):
        charges.append([0,slots[i][0],slots[i][1]])
    return charges

def set_single_negative_last(slots):
    q = 1.0/(len(slots)-1)
    charges = [[0, i[0],i[1]] for i in slots]
    charges[-1][0] = -q
    return charges 

def get_field_centre(charges):
    net_field = np.array([0.0,0.0])
    for charge in charges:
        r = np.array([charge[1],charge[2]])
        rabs = np.linalg.norm(np.array(charge[1:]))
        q = charge[0]
        net_field += -q * r / rabs**3
    return net_field

def plot_charges(charges,field,saveas="random.png",arrow_scale=0.5):
    # green is neutral, red is positive, blue is negative
    neutral,positive,negative = [],[],[]
    for charge in charges:
        if charge[0]==0:
            neutral.append([charge[1],charge[2]])
        elif charge[0]>0:
            positive.append([charge[1],charge[2]])
        elif charge[0]<0:
            negative.append([charge[1],charge[2]])
    neutral = np.array(neutral).T
    positive = np.array(positive).T
    negative = np.array(negative).T
    plt.figure(figsize=(8,8))
    if len(neutral): plt.plot(neutral[0],neutral[1],'go',label="neutral")
    if len(positive): plt.plot(positive[0],positive[1],'rx',label="positive charge")
    if len(negative): plt.plot(negative[0],negative[1],'bx',label="negative charge")
    
    origin = [0], [0]
    plt.quiver(*origin, field[0], field[1], color='k', scale=arrow_scale,label="field = {}\nnorm = {}".format(field,round(np.linalg.norm(np.array(field)),5)))
    plt.legend()
    
    plt.savefig(saveas)
    plt.close()

def part1():
    # Question 1, set 12 charges in 13 slots find the field. 
    slots1 = set_slots(13)
    charges1 = set_charges(12,slots1)
    field1 = get_field_centre(charges1)
    plot_charges(charges1 , field1 , saveas="12positive_charges.png") 
    print("\nWe set 12 out of 13 charges and the field at the centre is {}\n the strenth (absolute value) of the field at the centre is {}\n\n".format(field1,np.linalg.norm(field1)))
    # Set 1 charge q=-1/12 in the last of 13 slots, find the field
    charges2 = set_single_negative_last(slots1)
    field2 = get_field_centre(charges2)
    plot_charges(charges2,field2,saveas="1negative_charge.png") 
    print("Now we set one negative charge in the place where there was a hole,\nthe resultant field at the centre is {}\n the absolute value of this field is {}\n\n".format(field2 , np.linalg.norm(field2)))




def main():
    ### PART 1
    part1()

    ### PART 2
    field_array = []
    abs_field_array = []
    narr = [n for n in range(3,500,2)]
    for n in narr:
        m = int((n-1)/2)
        slots = set_slots(n)
        charges = set_charges(m,slots)
        field = get_field_centre(charges)
        plot_charges(charges , field , saveas="./bunch_of_charges/n={}.png".format(1000+n), arrow_scale=3)
        field_array.append(field)
        abs_field_array.append(np.linalg.norm(np.array(field)))
    plt.figure(figsize=(8,8))
    plt.plot(narr , abs_field_array)
    plt.xlabel("n")
    plt.ylabel("net force field abs value")
    plt.savefig("force_field_with_n.png")
    plt.show()

main()
