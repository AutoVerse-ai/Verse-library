import matplotlib.pyplot as plt 
import xml.etree.ElementTree as ET

if __name__ == "__main__":
    fn = 'ARG_Carcarana-11_1_T-1.xml'
    tree = ET.parse(fn)
    root = tree.getroot()

    lanelet_list = root.findall('lanelet')
    for j in range(len(lanelet_list)):
        example_lanelet = lanelet_list[j]
        left_bound = example_lanelet.find('leftBound')
        right_bound = example_lanelet.find('rightBound')
        leftpoint_list = left_bound.findall('point')
        rightpoint_list = right_bound.findall('point')
        assert(len(leftpoint_list)==len(rightpoint_list))
        left_x = []
        left_y = []
        right_x = []
        right_y = []
        middle_x = []
        middle_y = []
        for i in range(len(leftpoint_list)):
            leftpoint = leftpoint_list[i]
            left_x.append(float(leftpoint.find('x').text))
            left_y.append(float(leftpoint.find('y').text))

            rightpoint = rightpoint_list[i]
            right_x.append(float(rightpoint.find('x').text))
            right_y.append(float(rightpoint.find('y').text))
        
            middle_x.append((float(leftpoint.find('x').text)+float(rightpoint.find('x').text))/2)
            middle_y.append((float(leftpoint.find('y').text)+float(rightpoint.find('y').text))/2)

        # plt.plot(left_x, left_y,'-*')
        # plt.plot(right_x, right_y,'-*')
        plt.plot(middle_x, middle_y, '-*')

    plt.show()