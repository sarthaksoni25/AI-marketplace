import os
import cv2
import sys
import numpy as np
import random
from operator import itemgetter
import scipy.stats as stats
from scipy.stats import truncnorm

# def get_q(a,b,mu,sigma):
#     #a, b = 0, 1

#     dist = stats.truncnorm((a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)

#     values = dist.rvs(1)
#     return round(values[0],3)

def get_q(low,upp,mu,sigma):
    s = np.random.normal(mu, sigma, 1)
    if(s[0]>=low and s[0]<=upp):
      return s[0]
    else:
      return (low+upp)/2

#function to create Agent 1
def create_agent1(ID,Reputation,mu,sig,mup,sigp):
    Revenue=[]
    Datasets=[]
    #print("ID:",ID,"Reputation:",Reputation,"mu,sig:",mu,sig)

    return [ID,Reputation,Datasets,Revenue,mu,sig,mup,sigp]

#function to create Agent 2
def create_agent2(ID,TYPE,price,k,mu,sig):
    a=round(random.random(),2)
    b=round(random.random(),2)
    g=round(random.random(),2)
    t=random.uniform(0.0,0.1)
    sell_list=[]
    for j in range(k):
      sell_list.append(0)
    #v=t.rvs()
    #print("ID:",ID,"TYPE:",TYPE,"price:",price,"a,b:",a,b,"t:",t)

    #return [ID,TYPE,price,a,b,round(t,3)]
    # id , type, price , alpha, beta ,
    return [ID,TYPE,price,a,b,t,sell_list,mu,sig,g]

#function to create dataset

def create_dataset(parent,ID,TYPE,price,Quality,LID,flag):
    #print("parent:",parent,"ID:",ID,"TYPE:",TYPE,"price:",price,"Quality:",Quality)
    return [parent,ID,TYPE,price,Quality,0,0.0,LID,flag]

#function to create population of Agent 1

def population_agent1(N):
    
    pop = []
    # step=(1/N)
    for i in range(N):
        l=(1/N)*i
        r=(1/N)*(i+1)
        mu= random.uniform(l,r)
        sigma=random.uniform(0,0.1)
        r = get_q(0,1,mu,sigma)
        agent = create_agent1(i+1001, r,mu,sigma,mu,sigma) #create agent 1 with given id and reputation
        #print("seller:",i,"mu:",mu,"sigma:",sigma)
        pop.append(agent)

    return pop

#function to create population of Agent 2

def population_agent2(N,k):
    
    pop = []
    

    for i in range(N):
        
        t=random.randint(0,3) #randomly assign type of dataset the agent is looking for
        mu=random.uniform(0,1)
        sig=random.uniform(0,0.2)
        p=get_q(0,1,mu,sig) #randomly chose a price
        
        agent = create_agent2(i+2001, t,p,k,mu,sig) #create the agent with required id
        #print("Buyer:",i,"tol:",agent[5])
        pop.append(agent)

    return pop

#function to create population of datasets

def populate_datasets(Agent1,n,regulation,regulation_flag):
    data=[[],[],[],[]] #list of all present datasets according to their types
    for i in range(len(Agent1)):
      for j in range(n):
        # Q = [] #reputation vector
        # for x in range(7):
        Q=get_q(0,1,Agent1[i][4],Agent1[i][5]) # assign a quality to dataset based on mu,sig of seller
        o_mu=Agent1[i][4]
        o_sig=Agent1[i][5]  
        # print (float(sys.argv[1]))
        #v=Q.rvs()
        #d=create_dataset(Agent1[i][0],len(Agent1[i][2])+1,random.randint(1,4),round(random.random(),2),round(Q,3))
        typ=random.randint(0,3)
        # d=create_dataset(Agent1[i][0],len(Agent1[i][2])+1,typ,random.uniform(0,1),Q,len(data[typ]))
        if(Q < o_mu - regulation * o_sig):
          Agent1[i][1] = Agent1[i][1] / 3
          # d=create_dataset(Agent1[i][0],len(Agent1[i][2])+1,typ,Q,Q,len(data[typ]),regulation_flag)
        # else:
        d=create_dataset(Agent1[i][0],len(Agent1[i][2])+1,typ,Q,Q,len(data[typ]),False)
        # if(Q < regulation):
        #   d=create_dataset(Agent1[i][0],len(Agent1[i][2])+1,typ,Q,Q,len(data[typ]),regulation_flag)
        # else:
        #   d=create_dataset(Agent1[i][0],len(Agent1[i][2])+1,typ,Q,Q,len(data[typ]),False)
        
        #print("len(data[typ-1]):",len(data[typ-1]))
        #print("typ-1",typ-1)
        Agent1[i][3].append(0) #initialise the revenue earned from that dataset to be 0
        Agent1[i][2].append(d) #associate the dataset with its respective agent 1 by including it in agent 1s dataset list
        data[d[2]].append(d) #add the dataset appropriately in the market datasets list
        # print("d[2]-1:",d[2]-1)
        # print("after adding len(data[typ-1]):",len(data[typ-1]))
    # for x in range(len(data)):
    #   print(data[x])
    return data

def find(seller,agent,Dataset,typ):

    #storing the price,alpha and beta values of agent 2
    dlist=[]
    y = 1
    dlist.extend(Dataset[typ])
    # dlist=Dataset[typ]

    # print ("-----------------")
    # print (dlist[0][8]*y)
    # print ("-----------------")
    p=agent[2]
    a=agent[3]
    b=agent[4]
    g=agent[9]
    if(a>=b):
      dlist.sort(key=itemgetter(4), reverse=True) #sorting the dataset list according to quality
    else:
      dlist.sort(key=itemgetter(3), reverse=False) #sorting the dataset list according to price

    for i in range(len(dlist)):
      if (dlist[i][3]*b + dlist[i][4]*a - seller[dlist[i][0]-1001][1]*g - dlist[i][8]*y )/2<=p: #check if any dataset meets the agent 2 criteria
        return dlist[i]

    return []

# def find(seller,agent,Dataset,typ):
#     # print("before sorting:")
#     # for x in range(len(Dataset)):
#     #       print(Dataset[x])   

#     dlist=[] 
#     dlist.extend(Dataset[typ])
#     #storing the price,alpha and beta values of agent 2
#     temp=[]
#     p=agent[2]
#     a=agent[3]
#     b=agent[4]
#     g=agent[9]
#     if(a>=b):
#       dlist.sort(key=itemgetter(4), reverse=True) #sorting the dataset list according to quality
#     else:
#       dlist.sort(key=itemgetter(3), reverse=False) #sorting the dataset list according to price
#     for i in range(len(dlist)):
#       if (dlist[i][3]*b + dlist[i][4]*a - seller[dlist[i][0]-1001][1]*g)/2<=p: #check if any dataset meets the agent 2 criteria
#         temp.append(dlist[i])

#     #dlist.sort(key=itemgetter(4), reverse=True) #sorting the dataset list according to quality
#     loyalty=-1
#     ans=[]
#     for i in range(len(temp)):
#       if (agent[6][temp[i][0]-1001]>loyalty): #check if any dataset meets the agent 2 criteria
#         loyalty=agent[6][temp[i][0]-1001]
#         ans=temp[i]

#     # print("after sorting:")
#     # for x in range(len(Dataset)):
#     #     print(Dataset[x])    
#     return ans
def update_quality(Agent1,Dataset,regulation,regulation_flag):
  for i in range(len(Agent1)):      
        o_mu=Agent1[i][6]
        o_sig=Agent1[i][7]  
        for j in Agent1[i][2]:
          if(j[8]):
            # print ('j')
            # print (j)
            j[4] = o_mu - regulation*o_sig
            j[8] = False
            id = j[1]
            price = j[3]
            for k in Dataset[j[2]]:
              if (k[1] == id and k[3] == price):
                k[5] = o_mu - regulation*o_sig                
                k[8] = False
                # print('k')
                # print (k)
    # for x in range(len(Dataset)):
    #     print(Dataset[x])    
  return Agent1,Dataset

  
def intro_newdatasets(Agent1,Dataset,t,regulation,regulation_flag):
  #data=Dataset #list of all present datasets according to their types
  #Agent1=A1
  for i in range(len(Agent1)):      
      ans=0.0
      repu=Agent1[i][1]
      price=0.0
      #print(count)
      # for j in range(len(Agent1[i][2])):
      #     if(ans<Agent1[i][3][j]):
      #       ans=Agent1[i][3][j]
      #       t=Agent1[i][2][j][2]
      #       repu=Agent1[i][2][j][4]
      #       price=Agent1[i][2][j][3]
      # if(repu-0.2>=0):
      #   l=repu-0.2
      # else: 
      #   l=0
       
      t=random.randint(0,3)
      Q=get_q(repu,1,Agent1[i][6],Agent1[i][7]) # assign a quality
      o_mu=Agent1[i][6]
      o_sig=Agent1[i][7]  
      # print (float(sys.argv[1]))
      #Q=random.uniform(0,repu)
      #v=Q.rvs()
      #print("old repu:",repu)
      #d=create_dataset(Agent1[i][0],len(Agent1[i][2])+1,t,round(random.random(),2),round(Q,3))
      P=get_q(price,1,Agent1[i][4],Agent1[i][5])
      if(Q < o_mu - regulation * o_sig):
        Agent1[i][1] = Agent1[i][1]/3
        # d=create_dataset(Agent1[i][0],len(Agent1[i][2])+1,t,Q,Q,len(Dataset[t]),regulation_flag)
      # else:
      d=create_dataset(Agent1[i][0],len(Agent1[i][2])+1,t,Q,Q,len(Dataset[t]),False)
      # if(Q < regulation):
      #   d=create_dataset(Agent1[i][0],len(Agent1[i][2])+1,t,Q,Q,len(Dataset[t]),regulation_flag)
      # else:
      #   d=create_dataset(Agent1[i][0],len(Agent1[i][2])+1,t,Q,Q,len(Dataset[t]),False)
      
      Agent1[i][3].append(0) #initialise the revenue earned from that dataset to be 0
      Agent1[i][2].append(d) #associate the dataset with its respective agent 1 by inclusing it in agent 1s dataset list
      #print ("d[2]-1 = ",d[2]-1,test)
      Dataset[d[2]].append(d) #add the dataset appropriately in the market datasets list
  
  # for x in range(len(Dataset)):
  #     print(Dataset[x])    
  return Agent1,Dataset

def change_mu_sig1(Agents):
    Agent=Agents
    for i in range(len(Agent)):
      o_mu=Agent[i][4]
      o_sig=Agent[i][5]
      n_mu=random.uniform(o_mu,1)
      if(o_sig + 0.2<=1):
        r=o_sig + 0.2
      else:
        r=1  
      n_sig=random.uniform(o_sig,r)
      Agent[i][4]=n_mu
      Agent[i][5]=n_sig
    return Agent

def change_mu_sig2(Agents):
    Agent=Agents
    for i in range(len(Agent)):
      o_mu=Agent[i][4]
      o_sig=Agent[i][5]
      n_mu=random.uniform(0,o_mu)
      if(o_sig - 0.2>=0):
        r=o_sig - 0.2
      else:
        r=0  
      n_sig=random.uniform(r,o_sig)
      Agent[i][4]=n_mu
      Agent[i][5]=n_sig
    return Agent

def change_req(Agent2):
    A=Agent2
    for i in range(len(A)):
      A[i][1]=random.randint(0,3) #randomly assign type of dataset the agent is looking for
      A[i][2]=get_q(0,1,A[i][7],A[i][8]) #randomly chose a price
      # A[i][3]=round(random.random(),2)
      # A[i][4]=round(random.random(),2)
      # mu= round(random.random(),2)
      # sigma=round(random.random(),3)
      #tol=random.uniform(0,0.2)
      #A[i][5]=tol
      #A[i][3]=random.uniform(0,1)
      #A[i][4]=random.uniform(0,1)
      ##mu= random.random()
      ##sigma=random.random()
      ##A[i][5]=tol

    return A

def create_emptyfb(Agent1,s):
    empty_feedback=[]
    for a in range(len(Agent1)):
      empty_feedback.append([])
    for a in range(len(Agent1)):
      #print("creating empty feedback for seller:",a)
      for b in range(len(Agent1[a][2])):
        empty_feedback[a].append([Agent1[a][2][b][6],Agent1[a][2][b][5]])  
        #print(empty_feedback[a])  
  

    return empty_feedback

# def update_repu(Agent1,f,Dataset):
    
#     A=Agent1
#     D=Dataset
#     n=0
#     for i in range(len(f)):
#         s=0.0
#         for j in range(len(f[i])):
#           #print("Agent:",i,"score of dataset:",f[i][j][0],"purchase of Datasets:",f[i][j][1])
#           if(f[i][j][1]>0):
#             s+=f[i][j][0]/f[i][j][1]
#             D[A[i][2][j][2]-1][A[i][2][j][7]][5]=f[i][j][1]
#             D[A[i][2][j][2]-1][A[i][2][j][7]][6]=f[i][j][0]
#             D[A[i][2][j][2]-1][A[i][2][j][7]][4]=f[i][j][0]/f[i][j][1]
#             A[i][2][j][5]=f[i][j][1]
#             A[i][2][j][6]=f[i][j][0]
#             A[i][2][j][4]=f[i][j][0]/f[i][j][1]



#         r=s/(len(f[i]))
#         A[i][1]=r
#         print("seller:",i,"tot_score:",s,"Datasets:",len(A[i][2]),"len of f(i):",len(f[i]),"repu:",r)
      

#     return A,D


def update_repu(Agent1,feedback,Dataset):
    # print ("FEEDBACK")
    # print (feedback)
    # print ()
    #print_datasets(Agent1)
 
    for i in range(len(feedback)):
        s=0.0
        #print("seller:",i)
        for j in range(len(feedback[i])):
          if(feedback[i][j][1]>0):
            s+=feedback[i][j][0]/feedback[i][j][1]
            #print("seller:",i)
            #print("DS:",j,"score:",f[i][j][0],"count:",f[i][j][1])
            # print("typ:",Agent1[i][2][j][2])
            # print("Dataset:",Dataset[Agent1[i][2][j][2]])
            #print("update1:",Dataset[Agent1[i][2][j][2]][Agent1[i][2][j][7]])
            Dataset[Agent1[i][2][j][2]][Agent1[i][2][j][7]][5]=feedback[i][j][1]
            Dataset[Agent1[i][2][j][2]][Agent1[i][2][j][7]][6]=feedback[i][j][0]
            Dataset[Agent1[i][2][j][2]][Agent1[i][2][j][7]][4]=feedback[i][j][0]/feedback[i][j][1]
            #print("update2:",Agent1[i][2][j])
            Agent1[i][2][j][5]=feedback[i][j][1]
            Agent1[i][2][j][6]=feedback[i][j][0]
            #print_datasets(Agent1)
            Agent1[i][2][j][4]=feedback[i][j][0]/feedback[i][j][1]



        r=s/(len(feedback[i]))
        Agent1[i][1]=r
        #print("seller:",i,"tot_score:",s,"Datasets:",len(Agent1[i][2]),"len of f(i):",len(feedback[i]),"repu:",r)
      
    

    return Agent1,Dataset

def create_plot_repu(Agent1,plt_repu):

    plt=plt_repu
    for i in range(len(Agent1)):
        # plt[i].append(round(Agent1[i][1],3))
        plt[i].append(Agent1[i][1])

    return plt

def create_plot_data(count,c):

    plt=c
    for i in range(len(count)):
        plt[i].append(count[i])
    return plt

# def interact(A1,A2,f,cnt,Dataset,Card):
#   Agent1=A1
#   Agent2=A2
#   feedback=f
#   count=cnt
#   card=Card

#   for i in range(len(Agent2)): #ith Agent 2
#             t=Agent2[i][1]           #type of dataset agent 2 is looking for
#             #print ("trying...",i,"len:",len(Agent2))
#             ans=find(Agent1,Agent2[i],Dataset[t-1])  #find if any dataset which meets the agent's condition exist
#             if len(ans)>0 :
#               #score=round(random.random(),2)
#               Agent2[i][6][ans[0]-1001]+=1
#               score=get_q(0,1,Agent1[ans[0]-1001][4],Agent1[ans[0]-1001][5])
#               Agent1[ans[0]-1001][3][ans[1]-1]+=ans[3]
#               #print ("ans[0]-1001 = ",ans[0]-1001,"ans[1]-1 = ",ans[1]-1)
#               if(score>=ans[4]-Agent2[i][5] and score<=ans[4]):
#                 feedback[ans[0]-1001][ans[1]-1][0]+=ans[4] 
#               if(score<ans[4]-Agent2[i][5]):
#                 feedback[ans[0]-1001][ans[1]-1][0]+=score
#                 card[ans[0]-1001][-1]-=1
#               if( score>ans[4]):
#                 feedback[ans[0]-1001][ans[1]-1][0]+=score
#                 card[ans[0]-1001][-1]+=1
#               feedback[ans[0]-1001][ans[1]-1][1]+=1   
#               #print("score:",score,"diff limit:",ans[4]-Agent2[i][5])   
#               #print("buyer bought dataset from seller",ans[0]-1001,"with id:",ans[1]-1,"score:",feedback[ans[0]-1001][ans[1]-1][0],"purc:",feedback[ans[0]-1001][ans[1]-1][1])
#               count[ans[2]-1]+=1
#               print("seller:",ans[0]-1001,"dataset:",ans[1]-1,"score:",score)
#               print(feedback[ans[0]-1001])
#               #print(feedback[ans[0]-1001][ans[1]-1][1])
#               print(Agent2[i][6])
#   print("feedback before return:")
#   for x in range(len(feedback)):
#             print(feedback[i])
#   return Agent1,feedback,count,card

def interact(Agent1,Agent2,feedback,count,Dataset,Card,dcount):
  
  
  card=Card
  quality_flagged = 0
  quality_not_flagged = 0
  flagged = 0
  not_flagged = 0
  for i in range(len(Agent2)): #ith Agent 2
    t=Agent2[i][1]           #type of dataset agent 2 is looking for
    #print ("trying...",i,"len:",len(Agent2))
    ans=find(Agent1,Agent2[i],Dataset,t)  #find if any dataset which meets the agent's condition exist
    if len(ans)>0 :
      if(ans[8]):
        quality_flagged = quality_flagged + ans[4]
        flagged = flagged + 1
      else:
        quality_not_flagged = quality_not_flagged + ans[4]
        not_flagged = not_flagged + 1
      #score=round(random.random(),2)
      #print("interact seller:",ans[0]-1001)
      #Agent2[i][6][ans[0]-1001]+=1
      card[ans[0]-1001]+=1
      score=get_q(0,1,Agent1[ans[0]-1001][4],Agent1[ans[0]-1001][5])
      Agent1[ans[0]-1001][3][ans[1]-1]+=ans[3]  
      #print ("ans[0]-1001 = ",ans[0]-1001,"ans[1]-1 = ",ans[1]-1)
      if(score>=ans[4]-Agent2[i][5] and score<=ans[4]):
        feedback[ans[0]-1001][ans[1]-1][0]+=ans[4] 
        Agent2[i][6][ans[0]-1001]+=1
      if(score<ans[4]-Agent2[i][5]):
        feedback[ans[0]-1001][ans[1]-1][0]+=score
      if( score>ans[4]):
        feedback[ans[0]-1001][ans[1]-1][0]+=score
        Agent2[i][6][ans[0]-1001]+=1
      feedback[ans[0]-1001][ans[1]-1][1]+=1   
      #print("score:",score,"diff limit:",ans[4]-Agent2[i][5])   
      #print("buyer bought dataset from seller",ans[0]-1001,"with id:",ans[1]-1,"score:",f[ans[0]-1001][ans[1]-1][0],"purc:",f[ans[0]-1001][ans[1]-1][1])
      count[ans[2]]+=1
      dcount[ans[2]]+=1
      # print("seller:",ans[0]-1001,"dataset:",ans[1]-1,"score:",score)
      # print(f[ans[0]-1001])
      # print("feedback in after loop",i)
      # print(f)
      #print(feedback[ans[0]-1001][ans[1]-1][1])
      #print(Agent2[i][6])
  # print("feedback before return:")
  # print(f)
  # print ("----------------------------------")
  # print(q/sum(count))
  # print ("----------------------------------")
  if(flagged == 0):
    return Agent1,feedback,count,card,dcount, 0 , quality_not_flagged/not_flagged
  if(not_flagged == 0):
    return Agent1,feedback,count,card,dcount, quality_flagged/flagged , 0
  return Agent1,feedback,count,card,dcount, quality_flagged/flagged , quality_not_flagged/not_flagged
  

def print_datasets(Agent1):

  for i in range(len(Agent1)):
    print(Agent1[i][2])

def simulate(Agent1,Agent2,ns,nb, k,dcount,regulation,regulation_flag):
    
    # Agent1=population_agent1(ns)
    # Agent2=population_agent2(nb,ns)
    Dataset=populate_datasets(Agent1,10,regulation,regulation_flag)
    # print (Dataset)
    # print ("======================================================")
    Seller_card=[]
    #print("Agent1:",Agent1)
    # print("Initial Population:", population)
    
    # proportion = [] # make an empty list to keep track of the porportions after every interaction
    c=[[],[],[],[]]
    plt_repu=[]
    #dcount=[0,0,0,0]
    for i in range(len(Agent1)):
        plt_repu.append([])
        Seller_card.append(0)
        #c.append([])
    #empty_count=[0,0,0,0]
    count=[0,0,0,0]
    avg_count=[0.0,0.0,0.0,0.0]
    avg_quality_f = list()
    avg_quality_notf = list()
    for s in range(k):    #sth simulation
        if s==0:
            h=1
            plt_repu=create_plot_repu(Agent1,plt_repu)
        if s>0:
            Agent1,Dataset=intro_newdatasets(Agent1,Dataset,t_para,regulation,regulation_flag)   
            Agent2=change_req(Agent2)
        
        # print_datasets(Agent1)
        feedback=create_emptyfb(Agent1,s)
        Agent1,feedback,count,Seller_card,dcount,q_flagged,q_not_flagged=interact(Agent1,Agent2,feedback,count,Dataset,Seller_card,dcount) 
        # print("Agent1:",Agent1)
        # print("feedback:",feedback)
        # print("Dataset:",Dataset)
        avg_quality_f.append(q_flagged)
        avg_quality_notf.append(q_not_flagged)
        update_quality(Agent1,Dataset,regulation,regulation_flag)
        Agent1,Dataset=update_repu(Agent1,feedback,Dataset)
           
        # print("feedback after return:")
        # print(feedback)
        
        plt_repu=create_plot_repu(Agent1,plt_repu)
        #print(count)
        
        avg_count=[s*e for e in avg_count] 
        avg_count=[round(avg_count[l]+count[l],2)for l in range(len(count))]
        c=create_plot_data(count,c)
        t_para=avg_count.index(max(avg_count))
        if(t_para==0): 
            t_para=1
        count=[0,0,0,0]
        
        #Agent1=change_mu_sig1(Agent1))
    reven=[]
    trev=0.0
    for rev in range(len(Agent1)):
      lrev=0.0
      for dsets in range(len(Agent1[rev][3])):
        lrev+=Agent1[rev][3][dsets]
      reven.append(lrev)
      trev+=lrev      
    # print (avg_quality_f)
    # print ()
    # print (avg_quality_notf)
    return plt_repu,c,Seller_card,dcount,reven,avg_quality_f,avg_quality_notf

def name(arr):
  tot=0.0
  for x in arr:
    tot+=x
  if((tot/len(arr))>=0.66):
    return "high reputation" 
  elif((tot/len(arr))<0.66 and (tot/len(arr))>=0.33):
    return "average reputation"
  else:
    return "low reputation"

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
mpl.rcParams['font.sans-serif'] = ['Source Han Sans TW', 'sans-serif']
mpl.rcParams['figure.dpi'] = 200
# Then, "ALWAYS use sans-serif fonts"
# plt.rcParams['font.family'] = "Helvetica"
def plot_fig(repu):
    plt.figure()
    # plt.title('Change in Reputation of Seller over time')
    plt.ylabel('Reputation of sellers',fontname="Arial")
    #plt.yticks(np.arange(0,1.01,0.01))
    plt.xlabel('Number of Interactions',fontname="Arial")
    plt.axis()
    #put plot in the notebook
    c=0
    linestyles = ['-', '--', '-.', ':']
    colors=['black','gray']
    for i in repu:
        # name="seller"+str(c)
        plt.plot(i,linestyle=linestyles[(c-1)%4],label=name(i),color=colors[int(c/4)])
        c+=1
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('regulation =' + sys.argv[2] + '.png')

def plot_line_x(avg_quality_f, avg_quality_notf):
  plt.plot(avg_quality_f,color='black', label = 'flagged')
  plt.plot(avg_quality_notf ,  label = 'not flagged')
  plt.ylabel('Average Quality')
  plt.xlabel('Iterations') 
  plt.legend()
  plt.savefig('Quality(regulation =' + sys.argv[2] + ').png')
  plt.show()


def plot_bar_x(card):
    # this is for plotting purpose
    plt.figure(figsize=(10,10))
    objects = []
    for i in range(len(card)):
      objects.append(i+1)
    y_pos = np.arange(1,len(card))
    

    plt.bar(y_pos,card, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Usage')
    plt.xlabel('Seller')
    plt.title('Purchase Stats')

    plt.show()

def dbar(card,reven):
    # t = np.arange(0.01, 10.0, 0.01)
    # data1 = np.exp(t)
    # data2 = np.sin(2 * np.pi * t)
    tot=sum(card)
    card = [(i/tot)*100 for i in card]
    tot=sum(reven)
    reven = [(i/tot)*100 for i in reven]
    fig, ax1 = plt.subplots()
    # y_pos = []
    # for i in range(len(card)):
    #   y_pos.append(i+1)
    y_pos = np.arange(len(card))  
    color = 'black'
    ax1.grid()
    ax1.set_xlabel('Sellers')
    ax1.set_ylabel('percentage of purchases(or)revenue', color='black')
    ax1.bar(y_pos, card, width=0.2,label='Volume',color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # plt.legend(loc='center left', bbox_to_anchor=(0.5, 0.5))     

    #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'gray'
    #ax2.set_ylabel('Total Revenue earned', color='black')  # we already handled the x-label with ax1
    #ax2.bar(y_pos+0.2, reven,width=0.2,label='Revenue', color=color)
    ax1.bar(y_pos+0.2, reven,width=0.2,label='Value', color=color)
    #ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # labels=['revenue','p']
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))     
    plt.show()

def pie(sizes,keyword):
    plt.figure()
    # theme = plt.get_cmap('copper')
    # ax1.set_prop_cycle("color", [theme(1. * i / len(sizes))
    #                          for i in range(len(sizes))])
    colors = plt.cm.plasma(np.linspace(0., 1., 5))

    labels=[]
    temp=[]
    temp.extend(sizes)
    temp.sort()
    for i in range(len(sizes)):
        if(sizes[i]==temp[0]):
            labels.append("low "+keyword)
        elif(sizes[i]==temp[1]):
            labels.append("average "+keyword) 
        else:  
            labels.append("high "+keyword) 
     
    #colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    # explode = (0, 0, 0, 0)  # explode 1st slice

    # Plot
    plt.pie(sizes,labels=labels,radius=5000,
    autopct='%1.1f%%', shadow=False, startangle=140,colors=colors)

    plt.axis('equal')
    # plt.imshow(p1,cmap='gray')
    plt.show()

sellers=5
buyers=500
regulation_flag = bool(sys.argv[1])
regulation = float(sys.argv[2])
# Agent1=population_agent1(sellers)
# Agent2=population_agent2(buyers,sellers)

# for i in range(len(Agent1)):
#   print(Agent1[i])

# for i in range(len(Agent2)):
#   print(Agent2[i])

sim=500
batch=1
Repu=[]
repu=[]
dcount=[0,0,0,0]
for i in range(sellers):
  repu.append([])
  for j in range(sim):
    repu[i].append(0)

Agent1=population_agent1(sellers)
Agent2=population_agent2(buyers,sellers)
# Dataset1=populate_datasets(Agent1,10) 
# Dataset2=[]
# Dataset2.extend(Dataset1)   
# D=[]
# D.append(Dataset1)
# D.append(Dataset2)
for i in range(batch):
  srepu,count,card,dcount,reven,avg_qf,avg_qnf=simulate(Agent1,Agent2,sellers,buyers, sim,dcount,regulation,regulation_flag)
  print ()
  plot_fig(srepu)
  #pie(card,"sales")
  #pie(reven,"revenue")
  # dbar(card,reven)
  #plot_fig(card)
#   Repu.append(srepu)
#   #print (repu)
# #print (Repu)

# #print(repu[0])
# for i in range(batch):
#   for j in range(sellers):
#     for k in range(sim):
#       repu[j][k]+=Repu[i][j][k]

# for i in range(sellers):
#   for j in range(sim):
#     repu[i][j]=repu[i][j]/batch

# print(repu[0])
# print(len(repu))
# print(len(repu[0]))
# plot_fig(repu)
print(dcount)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# import matplotlib.pyplot as plt

# #plt.figure(figsize=(20,20))
# plt.title('Reputation of Seller over time')
# plt.ylabel('Reputation')
# #plt.yticks(np.arange(0,1.01,0.01))
# plt.xlabel('Time [No. of simulations]')
# #put plot in the notebook
# for i in repu:
#     plt.plot(i)

# # and add some details to the plot

# plt.ylim(0,1)
# plt.show()

print(dcount)

plt.figure()
objects = []
for i in range(len(dcount)):
  objects.append(i+1)
y_pos = np.arange(4)

print (sum(avg_qnf) / len(avg_qnf))
# plot_line_x(avg_qf,avg_qnf)


# plt.bar(y_pos,dcount, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Usage')
# plt.title('Programming language usage')

# plt.show()

# data = [[5., 25., 50., 20.],
#   [4., 23., 51., 17.],
#   [6., 22., 52., 19.]]

# X = np.arange(4)
# plt.bar(X + 0.00, data[0], color = 'b', width = 0.25)
# plt.bar(X + 0.25, data[1], color = 'g', width = 0.25)
# plt.bar(X + 0.50, data[2], color = 'r', width = 0.25)

# plt.show()

# include customer loyalty
# vary mu and sigma of seller(or buyer)(and let other remain same)
# regulatory dimension:
# 2)let the buyer also flag if reputation falls below tolerance
# 1)have a regulatory quality declared-

#!apt install gnuplot

# print(Agent2)

"""varying mu and sigma:

1.   if mu,sig of seller is increased:
     


>>* the reputation of sellers increase

2.   if mu,sig of seller is decreased:
3.   if the datasets produced in the market are of lowered in quality (by delta) than the prev ones:   
>* the market still sustains
"""