#Import all the lane objects we are going to use for lane object identification################
from turtle import right
import numpy as np #Numpy library to do the calculations
import matplotlib.pyplot as plt #Matplotlib library to visualize the roads
from bs4 import BeautifulSoup #Beautiful Soup library for parsing data in python 
###############################################################################################

#Import all the lane objects needed to generate all the lane map objects for the controllers
from verse.map.lane import Lane
from verse.map.lane_map import LaneMap
from verse.map.lane_segment import *
###############################################################################################

###############################ASAM OPEN DRIVE PARSING FUNCTION################################
def file_parser(file_name):
    with open(file_name,'r') as f: #now we are going to read each line inside the file
        data = f.read()  #Read all the file
    soup = BeautifulSoup(data,'xml') #store the data into the soup
    return soup
###############################################################################################

#########Check if we can get all the lane data from this tag from the ASAM Open DRIVE File
def check_valid_side(side):
    lane_array = [] #lane array
    temp_lanes = [] #temporary lanes to append
    alpha_array = [] #array of alpha values
    if side is not None: #we want to check is the side we are traversing is NOT NONE
        temp_lanes = side.find_all('lane') #let's find all the left lane
        for tl in temp_lanes: #traverse each element in the left lane
            lane_array.append(tl['type']) #add each of the left segment to the temporary left array
            temp_alpha = tl.find('width') #width of the temp road.
            alpha_array.append(float(temp_alpha['a'])) #add the alpha value into the array
        return True,lane_array,alpha_array #return True if the side is valid
    return False,[],[] #otherwise return False

#This the function that condenses the 2d matrix into lane segments
def condense_matrix(drive_way_matrix,width_2d):
    lanes_to_return = [] #to store the lanes to return
    width_to_return = [] #to store the width between each lane
    cols = len(drive_way_matrix[0]) #number of columns
    drive_way_matrix = np.array(drive_way_matrix) #driveway 2d matrix to traverse
    width_2d = np.array(width_2d) #width 2d matrix to traverse
    for i in range(cols):
        lane_to_append = Lane("Lane"+str(i+1),drive_way_matrix[:,i].tolist()) #combine all the rows into one list for each column
        lanes_to_return.append(lane_to_append) #add each Lane object to the list we want to return
        mean = np.mean(width_2d[:,i]) #calculate the mean of all the space between the lanes
        width_to_return.append(mean) #add it to the width we want to return
    return lanes_to_return,width_to_return

######This is the function which traverses each road element in a given road#####################
def road_traverser(road):
    left_drive_way_2d = [] 
    left_width_2d = []
    right_drive_way_2d = []
    right_width_2d = []

    for elem in road: #traverse each road segment in the open drive file
        planview = elem.find('planView') #we are going to find the planview
        road_geom = planview.find_all('geometry') #then find all the geometry
        lane_types = elem.find('lanes') #let's find all the lanes

        ###################LEFT SIDE OF THE ROAD##############################################################
        left = lane_types.find('left') #find all the left side of the road
        left_valid, left_array,left_alpha_array = check_valid_side(left) #check the left side of the lane

        #################RIGHT SIDE OF THE ROAD################################################################
        right = lane_types.find('right') #find all the center side of the road
        right_valid, right_array,right_alpha_array = check_valid_side(right) #check the right side of the lane
        #######################################################################################################
       
        for rg in road_geom: #traverse each geometry for each road
            left_temp_drive = [] #temporary list to store driveway segments 
            left_temp_width_2d = []
            right_temp_drive = [] #temporary list to store driveway segments 
            right_temp_width_2d = []
            temp_x = float(rg['x']) #don't forget to cast this to float!
            temp_y = float(rg['y']) #don't forget to cast this float!
            hdg = float(rg['hdg']) #gets the heading of the road
            length = float(rg['length']) #gets the length of the road
            
            if (rg.line): #we want to check if this is the line segment
                xi = temp_x #x-coordinate starting point
                xf = temp_x + np.cos(hdg) * length #x-coordinate ending point
                yi = temp_y #y-coordinate starting point
                yf = temp_y + np.sin(hdg) * length #y-coordinate ending point

                ###########################NORMALIZATION PART##############################
                delta_x = xf - xi #find the difference between two x-coordinates
                delta_y = yf - yi #find the differences between two y-coordinates
                N_left = np.array([-delta_y,delta_x])/(np.sqrt(delta_x**2 + delta_y**2)) #Normal Vector for left side
                N_right = np.array([delta_y,-delta_x])/(np.sqrt(delta_x**2 + delta_y**2)) #Normal Vector for right side
                ##############################################################################

                ########################LEFT ARRAY#######################
                left_alpha_val = 0
                left_idx = 0
                assert (len(left_array) == len(left_alpha_array))
             
                if left_valid: 
                    for left_elem in left_array:  
                        if left_elem == 'sidewalk':
                            style = '-'
                            left_alpha_val += left_alpha_array[left_idx]
                        elif left_elem == 'driving':
                            style = '--'
                            left_alpha_val += 0.5*left_alpha_array[left_idx]
                            left_new_vect_i = np.array([xi,yi]) + left_alpha_val*N_left
                            left_new_vect_f = np.array([xf,yf]) + left_alpha_val*N_left
                            id = str(left_idx)
                            width = left_alpha_array[left_idx]
                            line_types = style
                            straight_to_append = StraightLane(id,left_new_vect_i,left_new_vect_f,width,line_types,False,20,0) #Straight (id, start vector, end vector, width, line type,forbidden,speed limit, priority)
                            left_temp_drive.append(straight_to_append)
                            left_temp_width_2d.append(left_alpha_val)
                            left_alpha_val -= 0.5*left_alpha_array[left_idx]
                            left_alpha_val += left_alpha_array[left_idx]
                            
                        left_idx +=1 #increment left index
                
                ###################RIGHT ARRAY#########################
                right_alpha_val = 0
                right_idx = 0
                assert (len(right_array) == len(right_alpha_array))
                if right_valid: 
                    for right_elem in right_array: 
                        if right_elem == 'sidewalk':
                            style = '-'
                            right_alpha_val += right_alpha_array[right_idx]
                            
                        elif right_elem == 'driving':
                            style = '--'
                            right_alpha_val += 0.5*right_alpha_array[right_idx]
                            right_new_vect_i = np.array([xi,yi]) + right_alpha_val*N_right
                            right_new_vect_f = np.array([xf,yf]) + right_alpha_val*N_right
                            id = str(right_idx) 
                            width = right_alpha_array[right_idx]
                            line_types = style
                            straight_to_append = StraightLane(id,right_new_vect_i,right_new_vect_f,width,line_types,False,20,0) #Straight (id, start vector, end vector, width, line type,forbidden,speed limit, priority)
                            right_temp_drive.append(straight_to_append)
                            right_alpha_val -= 0.5*right_alpha_array[right_idx]
                            right_alpha_val += right_alpha_array[right_idx]
                            
                        right_idx +=1

            else: #if this is not a line segment then this is a curve
                #IMPORTANT #NOTE_TO KEEP IN MIND OF: 
                #if curvature is negative:
                #then the center is at the right hand side of the vehicle in vehicle's frame
                curvature = float(rg.find('arc')['curvature']) #curvature
                radius = np.abs(1/curvature) #radius = 1/curvature
                arc_length = float(rg['length']) #arc-length
                
                ############WE ARE GOING TO DETERMINE THE VECTOR DIRECTION####################
                T = np.array([np.cos(hdg) , np.sin(hdg),0]) #tangent vector on the curved line
                N = np.array([0,0,1]) #normal vector on the curved line
                S = np.cross(N,T) #cross product of normal and tangent vectors
                xi = temp_x #initial x-coordinate
                yi = temp_y #initial y-coordinate 
                center = [] #center coordinates
                curvature_direction = [] #store the direction of each curvature

                if curvature < 0: #if the curvature is negative then it it CLOCKWISE
                    center = np.array([xi,yi]) - S[0:2]*radius #center is equal to the current point subtract the radius
                    curvature_direction.append(False)
                else: #if the curvature is positive then it is COUNTER-CLOCKWISE
                    center = np.array([xi,yi]) + S[0:2]*radius #center is equal to the current point add the radius
                    curvature_direction.append(True)
                
                x_diff = xi - center[0] #x_diff means the difference between the current x-coordinate and the center x-coordinate
                y_diff = yi - center[1] #y_diff means the difference between the current y-coordinate and the center y-coordinate
                initial_theta = np.arctan2(y_diff, x_diff) #initial theta is the angle between the y_diff and x_diff
                delta_theta = abs(arc_length/radius) #delta theta is the final angle of the curvature

                if curvature < 0: #if the curvature is negative
                    final_theta = initial_theta - delta_theta  #we want to go the opposite way so subtract delta theta from initial theta
                else: #this condition means that the curvvature is positive
                    final_theta = initial_theta + delta_theta  #we want to go COUNTER-CLOCKWISE so ADD the delta theta to initial theta

                #####THIS IS WHERE I AM GOING TO ADJUST EACH LANE SEGMENT OF THE CURVE
                #########################LEFT ARRAY#########################################################
                left_idx = 0
                assert (len(left_array) == len(left_alpha_array))
                temp_radius = radius
                if left_valid:
                    for left_elem in left_array:
                        for curve_dir in curvature_direction:  
                            if curve_dir == True:
                                if left_elem == 'driving':
                                    temp_radius -= 0.5*left_alpha_array[left_idx]
                                    id = str(left_idx)
                                    center_points = center
                                    start_phase = initial_theta
                                    end_phase = final_theta
                                    clockwise = False
                                    width = left_alpha_array[left_idx]
                                    style = '--'
                                    circle_to_append = CircularLane(id,center_points,temp_radius,start_phase,end_phase,clockwise,width,style,False,20,0)
                                    left_temp_drive.append(circle_to_append)
                                    left_temp_width_2d.append(width)                                    
                                    temp_radius += 0.5*left_alpha_array[left_idx]
                                    temp_radius -= left_alpha_array[left_idx]

                                elif left_elem == 'sidewalk':
                                    style = '-'
                                    temp_radius -= left_alpha_array[left_idx]
                            
                            else:
                                if left_elem == 'driving':
                                    temp_radius += 0.5*left_alpha_array[left_idx]
                                    id = str(left_idx)
                                    center_points = center
                                    start_phase = initial_theta
                                    end_phase = final_theta
                                    clockwise = True
                                    width = left_alpha_array[left_idx]
                                    style = '--'
                                    circle_to_append = CircularLane(id,center_points,temp_radius,start_phase,end_phase,clockwise,width,style,False,20,0)
                                    left_temp_drive.append(circle_to_append)
                                    left_temp_width_2d.append(width)      
                                    temp_radius -= 0.5*left_alpha_array[left_idx]
                                    temp_radius += left_alpha_array[left_idx] 
                                    
                                elif left_elem == 'sidewalk':
                                    style = '-'
                                    temp_radius += left_alpha_array[left_idx]
                           
                            left_idx +=1  #increment the left index
                
                #############################RIGHT ARRAY##############################################
                right_idx = 0
                assert (len(right_array) == len(right_alpha_array))
                temp_radius = radius
                if right_valid:
                    for right_elem in right_array:
                        for curve_dir in curvature_direction:
                            if curve_dir == True:
                                if right_elem == 'driving':
                                    temp_radius += 0.5*right_alpha_array[right_idx]
                                    id = str(right_idx)
                                    center_points = center
                                    start_phase = initial_theta
                                    end_phase = final_theta
                                    clockwise = False
                                    width = right_alpha_array[right_idx]
                                    style = '--'
                                    circle_to_append = CircularLane(id,center_points,temp_radius,start_phase,end_phase,clockwise,width,style,False,20,0)
                                    right_temp_drive.append(circle_to_append)
                                    temp_radius -= 0.5*right_alpha_array[right_idx]
                                    temp_radius += right_alpha_array[right_idx]

                                elif right_elem == 'sidewalk':
                                    style = '-'
                                    temp_radius += right_alpha_array[right_idx]
                                
                            else:
                                if right_elem == 'driving':
                                    temp_radius -= 0.5*right_alpha_array[right_idx]
                                    id = str(right_idx)
                                    center_points = center
                                    start_phase = initial_theta
                                    end_phase = final_theta
                                    clockwise = False
                                    width = right_alpha_array[right_idx]
                                    style = '--'
                                    circle_to_append = CircularLane(id,center_points,temp_radius,start_phase,end_phase,clockwise,width,style,False,20,0)
                                    right_temp_drive.append(circle_to_append)
                                    temp_radius += 0.5*right_alpha_array[right_idx]
                                    temp_radius -= right_alpha_array[right_idx]
                                    
                                elif right_elem == 'sidewalk':
                                    style = '-'
                                    temp_radius -= right_alpha_array[right_idx]
                            right_idx +=1  #increment the right index
            left_drive_way_2d.append(left_temp_drive)  
            left_width_2d.append(left_temp_width_2d)
            right_drive_way_2d.append(right_temp_drive)  
            right_width_2d.append(right_temp_width_2d)

    return left_drive_way_2d,left_width_2d,right_drive_way_2d,right_width_2d

# def temp_right(): #this is just a temporary function to populate the right side
#     dp = []
#     widths_2d = []
#     for i in range(22):
#         list = []
#         width_temp = []
#         for j in range(3):
#             id = str(-(j+1))
#             start_vector = [0,0]
#             end_vector = [10,10]
#             width = 10
#             line_types = '-'
#             to_append = StraightLane(id,start_vector,end_vector,width,line_types,False,20,0) #Straight (id, start vector, end vector, width, line type,forbidden,speed limit, priority)
#             list.append(to_append)
#             width_temp.append(width)
#         dp.append(list)
#         widths_2d.append(width_temp)
        
#     lanes_to_return = []
#     width_to_return = []
#     cols = len(dp[0])
    
#     drive_way_matrix = np.array(dp)
#     width_2d = np.array(widths_2d)
#     for i in range(cols):
#         lane_to_append = Lane("Lane"+str(-(i+1)),drive_way_matrix[:,i].tolist())
#         lanes_to_return.append(lane_to_append)
#         mean = np.mean(width_2d[:,i])
#         width_to_return.append(mean)
#     return lanes_to_return

#Function to generate lane data visualization while parsing the ASAM Open DRIVE file
def opendrive_map(file_name): 
    soup = file_parser(file_name) #call the file parsing function
    road = soup.find_all('road')[:-1] #first find all the road segments and ignore the last one
    
    left_drive_way_2d,left_width_2d,right_drive_way_2d,right_width_2d = road_traverser(road)
    left_lanes,left_widths = condense_matrix(left_drive_way_2d,left_width_2d) #condense matrix returns left side of the road along the width 2d matrix
    right_lanes,right_widths = condense_matrix(right_drive_way_2d,right_width_2d) #condense matrix returns the right side of the road along the width 2d matrix
    #right_lanes = temp_right() #just a temporary function to generate the right side of the road 

    total = left_lanes[::-1] + right_lanes #reverse the left lane order and then combine with the right lane list to create one long lane list
    completed_lanes = LaneMap(total) #create a lane map

    for i in range(len(total)-1,0,-1):
        completed_lanes.left_lane_dict[total[i].id] = [total[i-1].id] #connect each left side for each lane

    for i in range(len(total)-1):
        completed_lanes.right_lane_dict[total[i].id] = [total[i+1].id] #connect each right side for each lane

    return completed_lanes #return the connected lane object

if __name__ == '__main__':
    file_name1 = 'map_package/Maps/t1_triple/OpenDrive/t1_triple.xodr' #parse in first file
    file_name2 = 'map_package/Maps/t2_triple/OpenDrive/t2_triple.xodr' #parse in second file
    file_name3 = 'map_package/Maps/t3/OpenDrive/t3.xodr' #parse in third file
    file_name4 = 'map_package/Maps/t4/OpenDrive/t4.xodr' #parse in fourth file
    file_name5 = 'map_package/Maps/track5/OpenDrive/track5.xodr' #parse in fifth file
    file_list = [file_name1,file_name2, file_name3, file_name4, file_name5] #store the file names in an array
    lane_list = []

    for idx,value in enumerate(file_list):
        lane_list = opendrive_map(value)
        # plt.figure()
        # lane_visualizer(value,idx)
        # plt.savefig('full_road_geometry/figure'+str(idx)+'.png')
        # plt.show() 

    # for idx,value in enumerate(file_list):
    #     plt.figure()
    #     center_visualizer(value, idx)   
    #     plt.savefig('road_centerline_geometry/figure'+str(idx)+'.png')
    #     plt.show()