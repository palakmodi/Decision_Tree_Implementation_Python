# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 17:31:37 2016

@author: Palak
"""
import math
def entropy(data, goal):

    #Calculates the entropy of the given data set for the target attribute.
    values = {}
    dt_entropy = 0.0

    # Calculate the frequency of each of the values in the target atribute
    for row in data:
        if (values.has_key(row[goal])):
            values[row[goal]] += 1.0
        else:
            values[row[goal]] = 1.0

    # Calculate the entropy of the data for the target attribute
    for f in values.values():
        dt_entropy = dt_entropy+ (-f/len(data)) * math.log(f/len(data), 2) 
        
    return dt_entropy
    
def gain(data, attr, goal):

    #Calculates the information gain. The attribute with highest information gain is chosen as the root
    values = {}
    sub_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for row in data:
        if (values.has_key(row[attr])):
            values[row[attr]] += 1.0
        else:
            values[row[attr]] = 1.0

    # Calculate the sum of the entropy for each sub of rows weighted
    # by their probability of occuring in the training set.
    for val in values.keys():
        val_prob = values[val] / sum(values.values())
        data_sub = [row for row in data if row[attr] == val]
        sub_entropy = sub_entropy+ val_prob * entropy(data_sub, goal)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(data, goal) - sub_entropy)
            

def target_highest_freq(data, goal):
   
   #which row has the maximum frequency for the target value enjoy
    data = data[:]
    return any_highest_freq([row[goal] for row in data])

def any_highest_freq(lst):
   
    #Returns the item that appears most frequently in the given list.
    lst = lst[:]
    best_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > best_freq:
            most_freq = val
            best_freq = lst.count(val)
            
    return most_freq

def unique(lst):
    #To Remove the redundant values from the list
    lst = lst[:]
    unique_lst = []

    # Cycle through the list and add each value to the unique list only once.
    for item in lst:
        if unique_lst.count(item) <= 0 or 0> unique_lst.count(item) :
            unique_lst.append(item)
            
    # Return the list with all redundant values removed.
    return unique_lst

def get_values(data, attr):
    #to get the unique values of the chosen attribute
    data = data[:]
    return unique([row[attr] for row in data])

def choose_attribute(data, attributes, goal, fitness):
    #Choose attribute with highest information gain
 
    data = data[:]
    best_gain = 0.0
    best_attr = None

    for attr in attributes:
        gain = fitness(data, attr, goal)
        if (gain >= best_gain and attr != goal):
            best_gain = gain
            best_attr = attr
                
    return best_attr

def get_data(data, attr, value):

    #Returns a list of all the rows in <data> with the value of <attr> matching the given value.
    data = data[:]
    rtn_lst = []
    
    if not data:
        return rtn_lst
    else:
        row = data.pop()
        if row[attr] == value:
            rtn_lst.append(row)
            rtn_lst.extend(get_data(data, attr, value))
            return rtn_lst
        else:
            rtn_lst.extend(get_data(data, attr, value))
            return rtn_lst

def get_prediction(row, tree):

    #Get prediction according to the attribute values.
    if type(tree) == type("string"):
        return tree
    # Traverse the tree further until a leaf node is found.
    else:
        attr = tree.keys()[0]
        t = tree[attr][row[attr]]
        return get_prediction(row, t)


def create_decision_tree(data, attributes, goal, fitness_func):
    
    #Returns a new decision tree based on the training data
  
    data = data[:]
    vals = [row[goal] for row in data]
    default = target_highest_freq(data, goal)

 
    if not data or (len(attributes) - 1) <= 0:
        return default

    elif vals.count(vals[0]) == len(vals) or vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
       
        best = choose_attribute(data, attributes, goal,
                                fitness_func)

        # Create a new decision tree/node with the best attribute 
        tree = {best:{}}
        for val in get_values(data, best):
            subtree = create_decision_tree(
                get_data(data, best, val),
                [attr for attr in attributes if attr != best],
                goal,
                fitness_func)
            tree[best][val] = subtree

    return tree
