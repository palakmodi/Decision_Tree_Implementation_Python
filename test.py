# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 17:31:37 2016

@author: Palak
"""
from decisiontree import *
import sys
import re

#  To load the data file which is inout by user
def getdatafile():
    print "Input the data file name: ", 
    filename= sys.stdin.readline().strip()
    file_ptr = open(filename, "r+")
    return file_ptr
    
# To read the data file into the file, in a key value pair using dictionary.
# After putting the data into dictionary, the create_decision_tree method is called.
def file_to_data(file_ptr):
    #dt = [line.strip() for line in file_ptr.readlines()]
   #data dt =[re.sub('[()\':;\n0-9]','',line) for line in file_ptr.readlines()]
    dt =[re.sub('[()\':;\n0-9]','',line) for line in file_ptr.readlines() if line.strip() != '']
    dt.reverse()
    attributes = [attr.strip() for attr in dt.pop().split(",")]
    target_attr = attributes[-1]
    dt.reverse()
    data = []
    for line in dt:
        data.append(dict(zip(attributes,
                             [datum.strip() for datum in line.split(",")])))
    # Create the decision tree
    tree = create_decision_tree(data, attributes, target_attr, gain)      
    return tree
    
    
def predict_in_tree(tree,target_attr):
    query={'Size': 'Large', 'Occupied': 'Moderate', 'Price':'Cheap', 'Music':'Loud', 'Location':'City-Center', 'VIP':'No', 'Favorite Beer':'No'}
    classification= get_prediction(query, tree)
    # Print out the prediction according to the query
    print query
    print "------------------------\n"
    print target_attr
    print "------------------------\n"
    print classification
    
    
def print_tree(tree, str):
    #To print the decision tree in the readable format
    if type(tree) == dict:
        print "%s%s" % (str, tree.keys()[0])
        for item in tree.values()[0].keys():
            print "%s\t%s" % (str, item)
            print_tree(tree.values()[0][item], str + "\t")
    else:
        print "%s\t->\t%s" % (str, tree)


if __name__ == "__main__":
    file_ptr = getdatafile()
    
    tree = file_to_data(file_ptr)
    print "\n"
    print "------------------------\n"
    print "--   Decision Tree    --\n"
    print "------------------------\n"
    print "\n"
    print_tree(tree, "")

    print "------------------------\n"
    print "--   Prediction   --\n"
    print "------------------------\n"
    print "\n"  
    predict_in_tree(tree,'Did you Enjoyed your night in Jerusalem?')
    file_ptr.close()
