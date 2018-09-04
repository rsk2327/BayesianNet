import pandas as pd
import numpy as np
import random
import copy

class Node(object):
    '''
    Defines a Node Class for storing characteristics and CPT of each node
    '''
    
    def __init__(self,name):
        self.parents = []
        self.children = []
        self.name = name
        self.cpt=[]
        self.limit = 3
        
    def addParent(self,x):
        self.parents.append(x)
    
    def addChild(self,x):
        self.children.append(x)
    
    def createCPT(self,data):
        cpt = computeProb(data,self.limit,self.parents,self.name)
        self.cpt = cpt


def computeProb(data,limit,cols,target):
    
    numCol = len(cols)
    
    if numCol==0:
        return(cpt_0(data,limit,cols,target))
    elif numCol ==1:
        return(cpt_1(data,limit,cols,target))
    elif numCol ==2:
        return(cpt_2(data,limit,cols,target))
    elif numCol ==3:
        return(cpt_3(data,limit,cols,target))
    else:
        return(cpt_4(data,limit,cols,target))
            

#Functions for computing the Conditional Probability Tables (CPTs)

def cpt_2(data,limit,cols,target):
    
    cpt = []
    alpha = 0.001
    
    for var1 in range(limit):
        for var2 in range(limit):
            
            totalN = len( data[ (data[cols[0]]==var1) & (data[cols[1]]==var2) ] )
            
            for targetVar in range(limit):
                
                count = len( data[ (data[cols[0]]==var1) & (data[cols[1]]==var2) & (data[target]==targetVar) ] )
                if totalN ==0:
                    cpt.append([var1,var2,targetVar, float(totalN + 3*alpha)])
                else:
                    cpt.append([var1,var2,targetVar, float(count)/float(totalN + 3*alpha)])
                    
    cpt = pd.DataFrame(cpt, columns=[cols[0],cols[1],target,'prob'])
                
    return(cpt)

def cpt_1(data,limit,cols,target):
    
    cpt = []
    alpha = 0.001
    
    for var1 in range(limit):
            
        
        totalN = len( data[ (data[cols[0]]==var1)] )
        
            
        for targetVar in range(limit):
            
            count = len( data[ (data[cols[0]]==var1) & (data[target]==targetVar) ] )
            
            if totalN ==0:
                cpt.append([var1,targetVar, float(totalN + 3*alpha)])
            else:
                cpt.append([var1,targetVar, float(count)/float(totalN + 3*alpha)])
                    
    cpt = pd.DataFrame(cpt, columns=[cols[0],target,'prob'])
                
    return(cpt)

def cpt_0(data,limit,cols,target):
    
    alpha = 0.001
    cpt = []
    
    
    totalN = len( data )
    
            
    for targetVar in range(limit):
            
        count = len( data[ (data[target]==targetVar) ] )
        if totalN ==0:
            cpt.append([targetVar, alpha/float(totalN + 3*alpha)])
        else:
            cpt.append([targetVar, float(count)/(totalN + 3*alpha)])
                    
    cpt = pd.DataFrame(cpt, columns=[target,'prob'])
                
    return(cpt)


def cpt_3(data,limit,cols,target):
    
    cpt = []
    alpha = 0.001
    
    for var1 in range(limit):
        for var2 in range(limit):
            for var3 in range(limit):
            
                totalN = len( data[ (data[cols[0]]==var1) & (data[cols[1]]==var2) & (data[cols[2]]==var3) ] )

                for targetVar in range(limit):

                    count = len( data[ (data[cols[0]]==var1) & (data[cols[1]]==var2) & (data[cols[2]]==var3) & (data[target]==targetVar) ] )
                    if totalN ==0:
                        cpt.append([var1,var2,var3,targetVar, alpha/float(totalN + 3*alpha)])
                    else:
                        cpt.append([var1,var2,var3,targetVar, float(count)/float(totalN + 3*alpha)])
                    
    cpt = pd.DataFrame(cpt, columns=[cols[0],cols[1],cols[2],target,'prob'])
                
    return(cpt)

def cpt_4(data,limit,cols,target):
    
    cpt = []
    alpha = 0.001
    
    for var1 in range(limit):
        for var2 in range(limit):
            for var3 in range(limit):
                for var4 in range(limit):
            
                    totalN = len( data[ (data[cols[0]]==var1) & (data[cols[1]]==var2) & (data[cols[2]]==var3) & (data[cols[3]]==var4) ] )

                    for targetVar in range(limit):

                        count = len( data[ (data[cols[0]]==var1) & (data[cols[1]]==var2) & (data[cols[2]]==var3) & (data[cols[3]]==var4) & (data[target]==targetVar) ] )
                        if totalN ==0:
                            cpt.append([var1,var2,var3,var4,targetVar, alpha/float(totalN + 3*alpha)])
                        else:
                            cpt.append([var1,var2,var3,var4,targetVar, float(count)/float(totalN + 3*alpha)])

    cpt = pd.DataFrame(cpt, columns=[cols[0],cols[1],cols[2],cols[3],target,'prob'])
                
    return(cpt)

structMap = {0:[1,2],1:[1,3],2:[1,4],3:[2,3],4:[2,4],5:[3,4]}   # Mapping of the structure position and the nodes that it connects


class BayesNet(object):
    
    def __init__(self,numNodes,structure):
        self.structure = structure  # Array that defines the structure of the Bayes Net
        self.numNodes = numNodes
        self.varNodes={}
        self.classNode=0
        
        
    def initGraph(self):
        '''
        Initializes components of the Bayes Net Graph
        '''
        
        self.classNode = Node('Class')
        
        for i in range(self.numNodes):
            self.varNodes['x'+str(i+1)]=Node('x'+str(i+1))
            self.varNodes['x'+str(i+1)].parents.append('Class')
        
        for i in range(len(self.structure)):
            
            edgeNodes = structMap[i]
            firstNode = 'x'+str(edgeNodes[0])
            secondNode = 'x'+str(edgeNodes[1])
            
            if self.structure[i]==1:
                self.varNodes[firstNode].children.append(secondNode)
                self.varNodes[secondNode].parents.append(firstNode)
            elif self.structure[i]==-1:
                self.varNodes[secondNode].children.append(firstNode)
                self.varNodes[firstNode].parents.append(secondNode)
    
    def compCPT(self,data):
        '''
        Computes Conditional Probability Table for all the nodes
        '''
        
        self.classNode.createCPT(data)
        
        for i in range(len(self.varNodes)):
            self.varNodes['x'+str(i+1)].createCPT(data)
            
    
    def predict(self,data):
        '''
        Predicts most likely class given a single data sample
        
        '''
        maxProb = 0
        maxProbClass = 0

        for classVal in range(3):

            dt = data.copy()
            dt["Class"] = classVal
            prob = 1.0

            for i in range(self.numNodes):
                #print('Node is x'+str(i+1))

                pt=self.varNodes['x'+str(i+1)].cpt

                mergeList = self.varNodes['x'+str(i+1)].parents + ['x'+str(i+1)]

                cpt_prob = pd.merge(left=pt,right=dt,on=mergeList,how='inner')['prob'][0]
                #print("cpt_prob is ",str(cpt_prob))

                prob = cpt_prob*prob

            #print("Class :%d Prob : %f"%(classVal,prob))

            if prob>maxProb:
                maxProb = prob
                maxProbClass = classVal
                
        return(maxProbClass)


            