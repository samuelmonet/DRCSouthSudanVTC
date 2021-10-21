import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import pickle
import pydeck as pdk
import re
from collections import Counter
from PIL import Image

#import variables

#########################  a faire #########################################
# 
#
###########################################################################"


#Variables Correl Description
#becomes 
#variable_x variable_y description


st.set_page_config(layout="wide")


#import des données
@st.cache
def load_data():
	data = pd.read_csv('viz.csv',sep='\t')
	data['flee_reason']=data['flee_reason'].apply(lambda x:'Returnee or Host' if x=='0' else x)
	
	return data

data=load_data()

#st.dataframe(correl)
#st.write(data.columns)
#st.write(correl.shape)

def sankey_graph(data,L,height=600,width=1600):
    """ sankey graph de data pour les catégories dans L dans l'ordre et 
    de hauter et longueur définie éventuellement"""
    
    nodes_colors=["blue","green","grey",'yellow',"coral"]
    link_colors=["lightblue","lightgreen","lightgrey","lightyellow","lightcoral"]
    
    
    labels=[]
    source=[]
    target=[]
    
    for cat in L:
        lab=data[cat].unique().tolist()
        lab.sort()
        labels+=lab
    
    for i in range(len(data[L[0]].unique())): #j'itère sur mes premieres sources
    
        source+=[i for k in range(len(data[L[1]].unique()))] #j'envois sur ma catégorie 2
        index=len(data[L[0]].unique())
        target+=[k for k in range(index,len(data[L[1]].unique())+index)]
        
        for n in range(1,len(L)-1):
        
            source+=[index+k for k in range(len(data[L[n]].unique())) for j in range(len(data[L[n+1]].unique()))]
            index+=len(data[L[n]].unique())
            target+=[index+k for j in range(len(data[L[n]].unique())) for k in range(len(data[L[n+1]].unique()))]
       
    iteration=int(len(source)/len(data[L[0]].unique()))
    value_prov=[(int(i//iteration),source[i],target[i]) for i in range(len(source))]
    
    
    value=[]
    k=0
    position=[]
    for i in L:
        k+=len(data[i].unique())
        position.append(k)
    
   
    
    for triplet in value_prov:    
        k=0
        while triplet[1]>=position[k]:
            k+=1
        
        df=data[data[L[0]]==labels[triplet[0]]].copy()
        df=df[df[L[k]]==labels[triplet[1]]]
        #Je sélectionne ma première catégorie
        value.append(len(df[df[L[k+1]]==labels[triplet[2]]]))
        
    color_nodes=nodes_colors[:len(data[L[0]].unique())]+["black" for i in range(len(labels)-len(data[L[0]].unique()))]
    #print(color_nodes)
    color_links=[]
    for i in range(len(data[L[0]].unique())):
    	color_links+=[link_colors[i] for couleur in range(iteration)]
    #print(L,len(L),iteration)
    #print(color_links)
   
   
    fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 30,
      line = dict(color = "black", width = 1),
      label = [i.upper() for i in labels],
      color=color_nodes
      )
      
    ,
    link = dict(
      source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
      target = target,
      value = value,
      color = color_links))])
    return fig

@st.cache
def count2(abscisse,ordonnée,dataf,title='',legendtitle='',xaxis=''):
    
    agg=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg2=agg.T/agg.T.sum()
    agg2=agg2.T*100
    agg2=agg2.astype(int)
    x=agg.index
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['coding'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        #st.write(labels,colors)
        for i in range(len(labels)):
            if labels[i] in data[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} %",textfont_color="black"))
        
    else:
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Persons','title_font':{'size':18}})
    fig.update_layout(legend_title=legendtitle,legend=dict(orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    fig.update_layout(title_text=title)
    
    return fig

@st.cache
def pourcent2(abscisse,ordonnée,dataf,title='',legendtitle='',xaxis=''):
    
    agg2=dataf[[abscisse,ordonnée]].groupby(by=[abscisse,ordonnée]).aggregate({abscisse:'count'}).unstack().fillna(0)
    agg=agg2.T/agg2.T.sum()
    agg=agg.T.round(2)*100
    x=agg2.index
    
    if ordonnée.split(' ')[0] in codes['list name'].values:
        colors_code=codes[codes['list name']==ordonnée.split(' ')[0]].sort_values(['coding'])
        labels=colors_code['label'].tolist()
        colors=colors_code['color'].tolist()
        fig = go.Figure()
        
        for i in range(len(labels)):
            if labels[i] in data[ordonnée].unique():
                fig.add_trace(go.Bar(x=x, y=agg[(abscisse,labels[i])], name=labels[i],\
                           marker_color=colors[i].lower(),customdata=agg2[(abscisse,labels[i])],textposition="inside",\
                           texttemplate="%{customdata} persons",textfont_color="black"))
        
    else:
        fig = go.Figure(go.Bar(x=x, y=agg.iloc[:,0], name=agg.columns.tolist()[0][1],marker_color='green'))
        for i in range(len(agg.columns)-1):
            fig.add_trace(go.Bar(x=x, y=agg.iloc[:,i+1], name=agg.columns.tolist()[i+1][1]))
    
    fig.update_layout(barmode='relative', \
                  xaxis={'title':xaxis,'title_font':{'size':18}},\
                  yaxis={'title':'Pourcentage','title_font':{'size':18}})
    fig.update_layout(legend_title=legendtitle,legend=dict(orientation='h',
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.01,font=dict(size=18),title=dict(font=dict(size=18))
    ))
    fig.update_layout(title_text=title)
    
    return fig




questions=pd.read_csv('questions.csv',sep='\t')
questions=questions[[i for i in questions.columns if 'Unnamed' not in i]]
codes=pd.read_csv('codes.csv',index_col=None,sep='\t').dropna(how='any',subset=['color'])
continues=pickle.load( open( "cont_feat.p", "rb" ) )
cat_cols=pickle.load( open( "cat_cols.p", "rb" ) )
dummy_cols=pickle.load( open( "dummy.p", "rb" ) )	
questions.set_index('Idquest',inplace=True)
correl=pd.read_csv('graphs.csv',sep='\t')
#st.write(questions)
text=[i for i in questions.columns if questions[i]['Treatment']=='text']
text2=[questions[i]['question'] for i in text if 'recomm' not in i]+['Recommandation progamming','Recommandation activities'] 
#st.write(text)

img1 = Image.open("logoAxiom.png")
img2 = Image.open("logoDRC.png")

def main():	
	
	
	st.sidebar.image(img1,width=200)
	st.sidebar.title("")
	st.sidebar.title("")
	topic = st.sidebar.radio('What do you want to do ?',('Display Machine Learning Results','Display correlations related to questions 37,38, 40 and 41','Display other correlations','Display Wordclouds'))
	
	title2,title3 = st.columns([5,2])
	title3.image(img2)
	
	#st.write(questions)
	#st.write(cat_cols)	
	if topic=='Display correlations related to questions 37,38, 40 and 41':
		
		quest=correl[correl['variable_y'].isin(['change income','change foodsec','change2 LH','change2 food_access'])].copy()
		title2.title('Correlations uncovered from the database:')
		title2.title('Focus on questions 37, 38, 40 and 41, related to feeling of improvement thanks to the project')
		
		
		for var in ['region','flee_reason','gender resp','course','marital']:
			
			st.markdown("""---""")
			st.header(questions[var]['question'])		
						
			if var=='region':
				st.write('If we look at the region of residence of the respondents:')
				st.write('We can see that according to all the 4 questions the project has been much more effective in Ajuong Thok than in Pamir')
			
					
				for correlation in quest[quest['variable_x']==var]['variable_y'].unique():
				
					st.subheader('Response to question: '+questions[correlation]['question'])
					col1, col2= st.columns([1,1])
					col1.plotly_chart(count2(var,correlation,data,xaxis='Region'),use_container_width=True)
					col2.plotly_chart(pourcent2(var,correlation,data,xaxis='Region'),use_container_width=True)
					st.write(correl[(correl['variable_x']==var) & (correl['variable_y']==correlation)]['description'].iloc[0])
				
			elif var=='flee_reason':
				st.write('If we look at the reason of displacement of the respondents:')
				st.write('We can see that all the host and returnees respondent believe that the project will help them on all the diferent aspect and 90% of the IDPs who fled because of lack of food. On the other hand, for those who fled for conflict reason this numbers drops to between 50 and 68% according to the question and to about 40% for those who fled for other reasons.')
				
				df=data[[var]+['change income','change foodsec','change2 LH','change2 food_access']].copy()
				
				for correlation in quest[quest['variable_x']==var]['variable_y'].unique():
					st.subheader('Response to question: '+questions[correlation]['question'])
					col1, col2= st.columns([1,1])
					col1.plotly_chart(count2(var,correlation,df,xaxis='Reason for displacement'),use_container_width=True)
					col2.plotly_chart(pourcent2(var,correlation,df,xaxis='Reason for displacement'),use_container_width=True)
					
			elif var=='gender resp':
				st.write('Male respondents tend to be slightly more confident in their future thanks to the project than women.')
			
					
				for correlation in quest[quest['variable_x']==var]['variable_y'].unique():
				
					st.subheader('Response to question: '+questions[correlation]['question'])
					col1, col2= st.columns([1,1])
					col1.plotly_chart(count2(var,correlation,data,xaxis='Sex of the respondent'),use_container_width=True)
					col2.plotly_chart(pourcent2(var,correlation,data,xaxis='Sex of the respondent'),use_container_width=True)
			
			elif var=='course':
				st.write('As observed with machine learning results it seems that the course of tailoring which is by far the most important one has reached its limits:')
				for correlation in quest[quest['variable_x']==var]['variable_y'].unique():
				
					st.subheader('Response to question: '+questions[correlation]['question'])
					col1, col2= st.columns([1,1])
					col1.plotly_chart(count2(var,correlation,data,xaxis='Course taken'),use_container_width=True)
					col2.plotly_chart(pourcent2(var,correlation,data,xaxis='Course taken'),use_container_width=True)
								
			
					
			else:
				st.write("Married respondents are more confident that the household's food security will improve because of the project than single and divorced respondents.")
			
					
				for correlation in quest[quest['variable_x']==var]['variable_y'].unique():
				
					st.subheader('Response to question: '+questions[correlation]['question'])
					col1, col2= st.columns([1,1])
					col1.plotly_chart(count2(var,correlation,data,xaxis='Marital status'),use_container_width=True)
					col2.plotly_chart(pourcent2(var,correlation,data,xaxis='Marital status'),use_container_width=True)
					
					
					
			
	
	elif topic=='Display other correlations':
		
		title2.title('Correlations uncovered from the database:')
		title2.title('Other questions')
		quest=correl[-correl['variable_y'].isin(['change income','change foodsec','change2 LH','change2 food_access'])].copy()
		st.write('')
		st.write('')
		st.write('')
		st.write('')
		k=0
		
		
		
		for i in range(len(quest)):
			
			st.markdown("""---""")		
			if quest.iloc[i]['variable_x'] in cat_cols or quest.iloc[i]['variable_y'] in cat_cols:
				
				if quest.iloc[i]['variable_x'] in cat_cols:
					cat,autre=quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']
				else:
					cat,autre=quest.iloc[i]['variable_y'],quest.iloc[i]['variable_x']
				#st.write('cat: ',cat,' et autre: ',autre)
						
				df=pd.DataFrame(columns=[cat,autre])
				
				catcols=[j for j in data.columns if cat in j]
				cats=[' '.join(i.split(' ')[1:])[:57] for i in catcols]
				
				for n in range(len(catcols)):
					ds=data[[catcols[n],autre]].copy()
					ds=ds[ds[catcols[n]]==1]
					ds[catcols[n]]=ds[catcols[n]].apply(lambda x: cats[n])
					ds.columns=[cat,autre]
					df=df.append(ds)
				df['persons']=np.ones(len(df))		
				#st.write(df)		
				
				#st.write(quest.iloc[i]['graphtype'])
						
				if quest.iloc[i]['graphtype']=='treemap':
					
					fig=px.treemap(df, path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']],\
					 values='persons')
					fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20))
					st.write(quest.iloc[i]['description'])
					st.plotly_chart(fig,use_container_width=True)
					k=0
					
					
				elif quest.iloc[i]['graphtype']=='sunburst':
					fig = px.sunburst(df.fillna(''), path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], values='persons',color=quest.iloc[i]['variable_y'])
					fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20))
					st.write(quest.iloc[i]['description'])
					st.plotly_chart(fig,use_container_width=True)
					k+=1
					
						
			else:	
				df=data[[quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y']]]
				df['persons']=np.ones(len(df))
				
				if quest.iloc[i]['graphtype']=='sunburst':
					fig = px.sunburst(df.fillna(''), path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], values='persons',color=quest.iloc[i]['variable_y'])
					fig.update_layout(title_text=quest.iloc[i]['variable_x'] + ' and ' +quest.iloc[i]['variable_y'],font=dict(size=20))
					st.plotly_chart(fig,size=1000)
				
				elif quest.iloc[i]['graphtype']=='treemap':
					fig=px.treemap(df, path=[quest.iloc[i]['variable_x'], quest.iloc[i]['variable_y']], values='persons')
					fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20))
					st.write(quest.iloc[i]['description'])
					st.plotly_chart(fig,use_container_width=True)
					
					k=0
				
					
				elif quest.iloc[i]['graphtype']=='violin':
					col1,col2=st.columns([1,1])
					fig = go.Figure()

					categs = data[quest.iloc[i]['variable_x']].unique()

					for categ in categs:
					    fig.add_trace(go.Violin(x=df[quest.iloc[i]['variable_x']][df[quest.iloc[i]['variable_x']] == categ],
                            			y=df[quest.iloc[i]['variable_y']][df[quest.iloc[i]['variable_x']] == categ],
                            			name=categ,
                            			box_visible=True,
                            			meanline_visible=True,points="all",))
					fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20),showlegend=False)
					fig.update_yaxes(range=[-0.1, df[quest.iloc[i]['variable_y']].max()+1],title=quest.iloc[i]['ytitle'])
					st.write(quest.iloc[i]['description'])
					st.plotly_chart(fig,use_container_width=True)
					
									
				elif quest.iloc[i]['graphtype']=='bar':
					st.write(quest.iloc[i]['description'])
					col1,col2=st.columns([1,1])

					fig1=count2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
					df,xaxis=quest.iloc[i]['xtitle'])
					fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20),showlegend=False)
					col1.plotly_chart(fig1,use_container_width=True)
					
					fig2=pourcent2(quest.iloc[i]['variable_x'],quest.iloc[i]['variable_y'],\
					df,xaxis=quest.iloc[i]['xtitle'])
					fig.update_layout(title_text=quest.iloc[i]['title'],font=dict(size=20),showlegend=False)
					col2.plotly_chart(fig2,use_container_width=True)
##############################################WORDCLOUDS##########################################################"						
						
	elif topic=='Display Wordclouds':
		
		x, y = np.ogrid[100:500, :600]
		mask = ((x - 300)/2) ** 2 + ((y - 300)/3) ** 2 > 100 ** 2
		mask = 255 * mask.astype(int)
	
		courses,child=False,False
		title2.title('Wordclouds for open questions')
		
		feature=st.sidebar.selectbox('Select the question for which you would like to visualize wordclouds of answers',[i for i in text2])	
		
		
		if 'Recommandation' not in feature:
			
			var=[i for i in questions if questions[i]['question']==feature][0]
		
			if var in ['type_business','profitable_explain','howmarket_functionning','business_difficulties']:
				df=data[data['running_business']=='Yes'].copy()
			else:
				df=data.copy()
			
			col1, col2, col3 = st.columns([1,4,1])
			col2.title('Wordcloud from question:')
			col2.title(feature)
				
		
			corpus=' '.join(df[var].apply(lambda x:'' if x=='0' else x))
			corpus=re.sub('[^A-Za-z ]',' ', corpus)
			corpus=re.sub('\s+',' ', corpus)
			corpus=corpus.lower()
		
			col3.title('')
			col3.title('')
			col3.title('')
			sw=col3.multiselect('Select words you would like to remove from the wordcloud \n\n', [i[0] for i in Counter(corpus.split(' ')).most_common() if i[0] not in STOPWORDS][:20])
		
			if corpus==' ':
	    			corpus='No_response'
			else:
				corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
		
			wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)		
			wc.generate(corpus)
			col2.image(wc.to_array(), use_column_width = True)	
		
			if col2.checkbox('Would you like to filter Wordcloud according to other questions'):
		
				feature2=col2.selectbox('Select one question to filter the wordcloud',[questions[i]['question'] for i in questions.columns if i not in text])		
				filter2=[i for i in questions if questions[i]['question']==feature2][0]
			
				if filter2 in continues:
					mini=int(data[filter2].fillna(0).min())
					maxi=int(data[filter2].fillna(0).max())
					minimum=col2.slider('Select the minimum value you want to visulize', min_value=mini,max_value=maxi)
					maximum=col2.slider('Select the maximum value you want to visulize', min_value=minimum,max_value=maxi+1)
					df=df[(df[filter2]>=minimum)&(df[filter2]<=maximum)]	
				
			
				else:
					filter3=col2.multiselect('Select the responses you want to include', [i for i in data[filter2].unique()])
					df=df[df[filter2].isin(filter3)]
			
				corpus=' '.join(df[var].apply(lambda x:'' if x=='0' else x))
				corpus=re.sub('[^A-Za-z ]',' ', corpus)
				corpus=re.sub('\s+',' ', corpus)
				corpus=corpus.lower()
			
				if corpus==' ' or corpus=='':
    					corpus='No_response'
				else:
					corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
		
				wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
				wc.generate(corpus)
				col2.image(wc.to_array(), use_column_width = True)
		
		
			if questions[var]['parent'] in questions.columns:
			
				child=True		
				var2=questions[var]['parent']
				st.markdown("""---""")	
				st.subheader('Wordclouds according to question : '+questions[var2]['question'])
				st.markdown("""---""")	
				if var2!='profitable':
					subcol1,subcol2=st.columns([1,1])
				else:
					subcol1,subcol2,subcol3=st.columns([1,1,1])
			
				L=df[var2].unique()
				
				corpus1=corpus2=corpus3=''
				Corpuses=[corpus1,corpus2,corpus3]
				
				for i in range(len(L)):		
			
					Corpuses[i]=' '.join(df[df[var2]==L[i]][var].apply(lambda x:'' if x=='0' else x))
					Corpuses[i]=re.sub('[^A-Za-z ]',' ', Corpuses[i])
					Corpuses[i]=re.sub('\s+',' ', Corpuses[i])
					Corpuses[i]=Corpuses[i].lower()
					if Corpuses[i]==' ':
    						Corpuses[i]='No_response'
					else:
						Corpuses[i]=' '.join([i for i in Corpuses[i].split(' ') if i not in sw])
					wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
					wc2.generate(Corpuses[i])
					if i==0:
						subcol1.write(str(L[i])+' : '+str(len(df[df[var2]==L[i]]))+' '+'repondents')
						subcol1.image(wc2.to_array(), use_column_width = True)
					elif i==1:
						subcol2.write(str(L[i])+' : '+str(len(df[df[var2]==L[i]]))+' '+'repondents')
						subcol2.image(wc2.to_array(), use_column_width = True)
					else:
						subcol3.write(str(L[i])+' : '+str(len(df[df[var2]==L[i]]))+' '+'repondents')
						subcol3.image(wc2.to_array(), use_column_width = True)
		
			subcol1,subcol2=st.columns([2,2])	
		
			if subcol1.checkbox('Would you like to filter Wordcloud according to courses followed?'):
			
				#st.write(data[data['running_business']=='Yes']['course'].value_counts())
				courses=True
			
				var3='course'
				st.markdown("""---""")	
				st.subheader('Wordclouds according to question : '+questions['course']['question'])
			
				list_courses=data[var3].unique()
			
				if child:
					if subcol2.checkbox('Include filter according to '+ questions[var2]['question']):
					#st.write(list_courses,L)			
						for i in range(4):		
							st.markdown("""---""")	
							sub1col1,sub1col2=st.columns([2,2])
							corpus1=corpus2=corpus3=corpus4=corpus5=corpus6=corpus7=corpus8=''					
							Corpuses=[corpus1,corpus2,corpus3,corpus4,corpus5,corpus6,corpus7,corpus8]	
						
							for k in range(2):
							
								Corpuses[4*k]=' '.join(df[df[var3]==list_courses[2*i+k]][var].apply(lambda x:'' if x=='0' else x))
								Corpuses[4*k]=re.sub('[^A-Za-z ]',' ', Corpuses[4*k])
								Corpuses[4*k]=re.sub('\s+',' ', Corpuses[4*k])
								Corpuses[4*k]=Corpuses[4*k].lower()
						
								if Corpuses[4*k]==' ' or Corpuses[4*k]=='':
    									Corpuses[4*k]='No_response'
								else:
									Corpuses[4*k]=' '.join([z for z in Corpuses[4*k].split(' ') if i not in sw])
													
							wc0 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
							wc0.generate(Corpuses[0])
							wc4 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
							wc4.generate(Corpuses[4])
							sub1col1.write(str(list_courses[2*i])+' : '+str(len(df[df[var3]==list_courses[2*i]]))+' '+'repondents')
							sub1col1.image(wc0.to_array(), use_column_width = True)
							sub1col2.write(str(list_courses[2*i+1])+' : '+str(len(df[df[var3]==list_courses[2*i+1]]))+' '+'repondents')
							sub1col2.image(wc4.to_array(), use_column_width = True)
							#st.write(Corpuses)
											
							for j in range(len(L)):
								if var2!='profitable':
									sub2col1,sub2col2,sub2col3,sub2col4=st.columns([1,1,1,1])
								else:
									sub2col1,sub2col2,sub2col5,sub2col3,sub2col4,sub2col6=st.columns([1,1,1,1,1,1])
							
								for k in range(2):
									#st.write(2*i+k,j,list_courses,L)
									Corpuses[4*k+1+j]=' '.join(df[(df[var2]==L[j]) & (df[var3]==list_courses[2*i+k])][var].apply(lambda x:'' if x=='0' else x))
									#st.write(Corpuses[4*k+1+j])
									Corpuses[4*k+1+j]=re.sub('[^A-Za-z ]',' ', Corpuses[4*k+1+j])
									Corpuses[4*k+1+j]=re.sub('\s+',' ', Corpuses[4*k+1+j])
									Corpuses[4*k+1+j]=Corpuses[4*k+1+j].lower()
							
									if Corpuses[4*k+1+j]==' ' or Corpuses[4*k+1+j]=='':
			   							Corpuses[4*k+1+j]='No_response'
									else:
										Corpuses[4*k+1+j]=' '.join([z for z in Corpuses[4*k+1+j].split(' ') if i not in sw])
						
							#st.write(Corpuses)
							wc1 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
							wc1.generate(Corpuses[1])				
							sub2col1.write(L[0]+' : '+str(len(df[(df[var3]==list_courses[2*i]) & (df[var2]==L[0])])))
							sub2col1.image(wc1.to_array(), use_column_width = True)
						
							wc5 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
							wc5.generate(Corpuses[5])				
							sub2col3.write(L[0]+' : '+str(len(df[(df[var3]==list_courses[2*i+1]) & (df[var2]==L[0])])))
							sub2col3.image(wc5.to_array(), use_column_width = True)
						
							wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
							wc2.generate(Corpuses[2])				
							sub2col2.write(L[1]+' : '+str(len(df[(df[var3]==list_courses[2*i]) & (df[var2]==L[1])])))
							sub2col2.image(wc2.to_array(), use_column_width = True)
						
							wc6 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
							wc6.generate(Corpuses[6])				
							sub2col4.write(L[1]+' : '+str(len(df[(df[var3]==list_courses[2*i+1]) & (df[var2]==L[1])])))
							sub2col4.image(wc6.to_array(), use_column_width = True)
						
							if var2=='profitable':
								wc3 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
								wc3.generate(Corpuses[3])				
								sub2col5.write(L[2]+' : '+str(len(df[(df[var3]==list_courses[2*i]) & (df[var2]==L[2])])))
								sub2col5.image(wc3.to_array(), use_column_width = True)
							
								wc7 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
								wc7.generate(Corpuses[7])				
								sub2col6.write(L[2]+' : '+str(len(df[(df[var3]==list_courses[2*i+1]) & (df[var2]==L[2])])))
								sub2col6.image(wc7.to_array(), use_column_width = True)			
					
			
				else:
					subcol1,subcol2=st.columns([2,2])
								
					corpus1=corpus2=corpus3=corpus4=corpus5=corpus6=corpus7=corpus8=''
					Corpuses=[corpus1,corpus2,corpus3,corpus4,corpus5,corpus6,corpus7,corpus8]
				
					for i in range(len(list_courses)):		
			
						Corpuses[i]=' '.join(df[df[var3]==list_courses[i]][var].apply(lambda x:'' if x=='0' else x))
						Corpuses[i]=re.sub('[^A-Za-z ]',' ', Corpuses[i])
						Corpuses[i]=re.sub('\s+',' ', Corpuses[i])
						Corpuses[i]=Corpuses[i].lower()
						if Corpuses[i]==' ' or Corpuses[i]=='':
	    						Corpuses[i]='No_response'
						else:
							Corpuses[i]=' '.join([i for i in Corpuses[i].split(' ') if i not in sw])
						wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
						wc2.generate(Corpuses[i])
						if i%2==0:
							subcol1.write(str(list_courses[i])+' : '+str(len(df[df[var3]==list_courses[i]]))+' '+'repondents')
							subcol1.image(wc2.to_array(), use_column_width = True)
						else:
							subcol2.write(str(list_courses[i])+' : '+str(len(df[df[var3]==list_courses[i]]))+' '+'repondents')
							subcol2.image(wc2.to_array(), use_column_width = True)	
	
		##########################################Traitement spécifique Recommandations#######################################################
		else:
			df=data.copy()
			col1, col2, col3 = st.columns([1,1,1])
			
			if feature=='Recommandation progamming':
				colonnes=['recomm1_VTC','recomm2_VTC','recomm3_VTC']
				st.title('Wordcloud from question:')
				st.title('39) Any recommendations for future improvement in livelihoods Programming/VTC?')
			else: 
				colonnes=['recomm1_activities','recomm2_activities','recomm3_activities']
				st.title('Wordcloud from question:')
				st.title('44) What recommendations would you propose to improve the project activities?')
			
			st.title('')
			st.title('')
			st.title('')
			
			corpus=' '.join(data[colonnes[0]].dropna())+\
				' '.join(data[colonnes[1]].dropna())+' '.join(data[colonnes[2]].dropna())
			corpus=re.sub('[^A-Za-z ]',' ', corpus)
			corpus=re.sub('\s+',' ', corpus)
			corpus=corpus.lower()
			sw=st.multiselect('Select words you would like to remove from the wordclouds \n\n', [i[0] for i in Counter(corpus.split(' ')).most_common() if i[0] not in STOPWORDS][:20])
			
			col1, col2, col3 = st.columns([1,1,1])
			
			for i in range(3):
				col_corpus=' '.join(data[colonnes[i]].dropna())
				col_corpus=re.sub('[^A-Za-z ]',' ', col_corpus)
				col_corpus=re.sub('\s+',' ', col_corpus)
				col_corpus=col_corpus.lower()
				if col_corpus==' ' or col_corpus=='':
		    			col_corpus='No_response'
				else:
					col_corpus=' '.join([i for i in col_corpus.split(' ') if i not in sw])		
				wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)		
				wc.generate(col_corpus)
				if i==0:
					col1.subheader('Recommandation 1')
					col1.image(wc.to_array(), use_column_width = True)
				elif i==1:
					col2.subheader('Recommandation 2')	
					col2.image(wc.to_array(), use_column_width = True)
				else:
					col3.subheader('Recommandation 3')
					col3.image(wc.to_array(), use_column_width = True)
					
			if st.checkbox('Would you like to filter Wordcloud according to other questions'):
				
						
				
				feature2=st.selectbox('Select one question to filter the wordcloud',[questions[i]['question'] for i in questions.columns if i not in text])		
				filter2=[i for i in questions if questions[i]['question']==feature2][0]
			
				if filter2 in continues:
					minimum=st.slider('Select the minimum value you want to visulize', 	min_value=data[filter2].fillna(0).min(),max_value=data[filter2].fillna(0).max())
					maximum=st.slider('Select the maximum value you want to visulize', min_value=minimum,max_value=data[filter2].fillna(0).max())
					df=df[(df[filter2]>=minimum)&(df[filter2]<=maximum)]	

				else:
					filter3=st.multiselect('Select the responses you want to include', [i for i in data[filter2].unique()])
					df=df[df[filter2].isin(filter3)]
				
								
				col1, col2, col3 = st.columns([1,1,1])
				for i in range(3):
					col_corpus=' '.join(df[colonnes[i]].dropna())
					col_corpus=re.sub('[^A-Za-z ]',' ', col_corpus)
					col_corpus=re.sub('\s+',' ', col_corpus)
					col_corpus=col_corpus.lower()
					if col_corpus==' ' or col_corpus=='':
		    				col_corpus='No_response'
					else:
						col_corpus=' '.join([i for i in col_corpus.split(' ') if i not in sw])		
					
					wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)		
					wc.generate(col_corpus)
					if i==0:
						col1.subheader('Recommandation 1')
						col1.image(wc.to_array(), use_column_width = True)
					elif i==1:
						col2.subheader('Recommandation 2')	
						col2.image(wc.to_array(), use_column_width = True)
					else:
						col3.subheader('Recommandation 3')
						col3.image(wc.to_array(), use_column_width = True)
				
				
			
			if st.checkbox('Would you like to filter Wordcloud according to courses followed?'):
				
				var3='course'
				
				st.markdown("""---""")	
				st.subheader('Wordclouds according to question : '+questions['course']['question'])
				list_courses=data[var3].unique()
					
				for i in range(len(list_courses)):		
					
					st.markdown("""---""")	
					st.write(str(list_courses[i])+' : '+str(len(df[df[var3]==list_courses[i]]))+' '+'repondents')
					col1, col2, col3 = st.columns([1,1,1])
					
					dft=df[df[var3]==list_courses[i]]
					
					for i in range(3):
						col_corpus=' '.join(dft[colonnes[i]].dropna())
						col_corpus=re.sub('[^A-Za-z ]',' ', col_corpus)
						col_corpus=re.sub('\s+',' ', col_corpus)
						col_corpus=col_corpus.lower()
						if col_corpus==' ' or col_corpus=='':
				    			col_corpus='No_response'
						else:
							col_corpus=' '.join([i for i in col_corpus.split(' ') if i not in sw])		
						wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)		
						wc.generate(col_corpus)
						if i==0:
							col1.subheader('Recommandation 1')
							col1.image(wc.to_array(), use_column_width = True)
						elif i==1:
							col2.subheader('Recommandation 2')	
							col2.image(wc.to_array(), use_column_width = True)
						else:
							col3.subheader('Recommandation 3')
							col3.image(wc.to_array(), use_column_width = True)
					
	elif topic=='Display Machine Learning Results':
		
		title2.title('Machine learning results on models trained on:')
		title2.title('Questions 37, 38, 40 and 41')
		
		
		st.title('')
		st.markdown("""---""")	
		st.subheader('Note:')
		st.write('A machine learning model has been run on the question related to feeling of improvement of the situation thanks to the project, the objective of this was to identify specificaly for these 4 questions. The models are run in order to try to predict as precisely as possible the feeling that the respondents expressed in their responses to these questions. The figures below show for each questions which parameter have a greater impact in the prediction of the model than a normal random aspect (following a statistic normal law')
		st.write('')
		st.write('Each line of the graph represent one feature of the survey that is important to predict the response to the question.')
		st.write('Each point on the right of the feature name represent one person of the survey. A red point represent a high value to the specific feature and a blue point a low value.')
		st.write('SHAP value: When a point is on the right side, it means that it contributed to a better note while on the left side, this specific caracter of the person reduced the final result of the prediction.')
		st.write('')
		st.write('The coding for the responses is indicated under the graph and the interpretation of the graphs is written below.')
		st.markdown("""---""")	
				
		temp = Image.open('changeincome.png')
		image = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image.paste(temp, (0, 0), temp)
		st.subheader('37) Since the DRC project my income has…')
		st.image(image, use_column_width = True)
		st.caption('Region o residence: Ajuong Thok:1 - Pamir:0')
		st.caption('How long have you been an IDP: Never:0 - 1-3 months:0,5 - 4-6 months:1 - About 1 year:2 - 1-2 years:3 - Over 2 years: 4')
		st.caption('Marital status: Married:1 - Single:0 - Widowed:0 - Divorced: 0')
		st.caption('Course Garnment making (tailoring): Has taken the course:1 - Did not take the course: 0')
		st.caption('Source of income: Land cultivation: Mentionnend Land cultivation as main source of income:1 - Did not mention land cultivation: 0')
		st.caption('')
		st.write('We can see that the main parameter for feeling that the level of income has increased is to live in Ajuong Thok. Which shows that the project has been much more effective in this region.')
		st.write('In second position comes the fact to be host or not displaced sine a long time and then to be married')
		st.write('Then other parameters could be to live in a small household and/or to have lower income per household member')
		st.write('It seems also that household with higher number of girls and household whose source of income is land cultivation also feel more confident in the fact that their level of income has incresased.')
		st.write('Finaly when we look at the courses, the tailoring course appears to be less effective for increasing the level of income than the other ones.')
		
		
		temp = Image.open('changefoodsec.png')
		image1 = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image1.paste(temp, (0, 0), temp)
		st.subheader('38) Since the DRC project the HHs food security has….')
		st.image(image1, use_column_width = True)		
		st.caption('Region o residence: Ajuong Thok:1 - Pamir:0')
		st.caption('How long have you been an IDP: Never:0 - 1-3 months:0,5 - 4-6 months:1 - About 1 year:2 - 1-2 years:3 - Over 2 years: 4')
		st.caption('Marital status: Married:1 - Single:0 - Widowed:0 - Divorced: 0')
		st.caption('Course Garnment making (tailoring): Has taken the course:1 - Did not take the course: 0')
		st.caption('Go to Ajuok Thok Marke for merchandises or spare parts: Go:1 - Do not go: 0')
		st.caption('')
		st.write('We find again the same conclusion as above.')
		st.write('On top of these, it seems that people who are going to Ajuok Thok Market (which further inforce the effectiveness of the project there) and people who have joined the VTC program since a significant time are more likely to believe that their food security has increased since the project started')
		
		
		
		temp = Image.open('change2LH.png')
		image2 = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image2.paste(temp, (0, 0), temp)
		st.subheader('40) Because of the project, I am confident that my livelihood will')
		st.image(image2, use_column_width = True)
		st.caption('Region o residence: Ajuong Thok:1 - Pamir:0')
		st.caption('How long have you been an IDP: Never:0 - 1-3 months:0,5 - 4-6 months:1 - About 1 year:2 - 1-2 years:3 - Over 2 years: 4')
		st.caption('Marital status: Married:1 - Single:0 - Widowed:0 - Divorced: 0')
		st.caption('Course Garnment making (tailoring): Has taken the course:1 - Did not take the course: 0')
		st.caption('')
		st.write('We observe the same main factors as above: region, household size, time since arrival, land cultivation as main income source and the fact not to have followed tailoring courses.')



		
		col1, col2, col3 = st.columns([1,4,2])
		temp = Image.open('change2food_sec.png')
		image3 = Image.new("RGBA", temp.size, "WHITE") # Create a white rgba background
		image3.paste(temp, (0, 0), temp)
		st.subheader('41) Because of the project, my household’s access to food will')
		st.image(image3, use_column_width = True)
		st.caption('Region o residence: Ajuong Thok:1 - Pamir:0')
		st.caption('How long have you been an IDP: Never:0 - 1-3 months:0,5 - 4-6 months:1 - About 1 year:2 - 1-2 years:3 - Over 2 years: 4')
		st.caption('Marital status: Married:1 - Single:0 - Widowed:0 - Divorced: 0')
		st.caption('Course Garnment making (tailoring): Has taken the course:1 - Did not take the course: 0')
		st.caption('')
		st.write('Here again the same aspects are coming out of the analyse')
		
		st.title('Conclusion/Recommandations')
		st.write('Given the outcomes of this analyse, I would advise:')  
		st.write('- Reduce the number of courses in tailoring')
		st.write('- Analyse differences between Ajuong Thok and Pamir to collect lessons learned and see how the success of Ajuong Thok could be replicated alsewhere.')
		st.write('- Understand why the project seems to be more effective with married people than with single or divorced people.')
		
		
		
		
		
		
		
		
		
	else:
		st.title('\t DRC South Sudan \t VTC')	


    
 
if __name__== '__main__':
    main()




    
