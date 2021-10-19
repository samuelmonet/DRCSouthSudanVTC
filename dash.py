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
	correl=pd.read_csv('graphs.csv',sep='\t')
	return data,correl

data,correl=load_data()

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




questions=pd.read_csv('questions.csv',index_col=None,sep='\t')
codes=pd.read_csv('codes.csv',index_col=None,sep='\t').dropna(how='any',subset=['color'])
continues=pickle.load( open( "cont_feat.p", "rb" ) )
cat_cols=pickle.load( open( "cat_cols.p", "rb" ) )
dummy_cols=pickle.load( open( "dummy.p", "rb" ) )	
questions.set_index('Idquest',inplace=True)



img1 = Image.open("logoAxiom.png")
img2 = Image.open("logoDRC.png")

def main():	
	
	
	st.sidebar.image(img1,width=200)
	st.sidebar.title("")
	st.sidebar.title("")
	topic = st.sidebar.radio('What do you want to do ?',('Display correlations related to questions 37,38, 40 and 41','Display other correlations','Display Wordclouds','Display Machine Learning Results'))
	
	title2,title3 = st.columns([5,2])
	title3.image(img2)
	
	#st.write(questions)
	#st.write(cat_cols)	
	if topic=='Display correlations related to questions 37,38, 40 and 41':
		
		quest=correl[correl['variable_y'].isin(['change income','change foodsec','change2 LH','change2 food_access'])].copy()
		title2.title('Correlations uncovered from the database:')
		title2.title('Focus on questions 37, 38, 40 and 41, related to feeling of improvement thanks to the project')
		
		
		for var in ['region','flee_reason','gender resp','marital']:
			
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
						
						
	elif topic=='Display Wordclouds':
		title2.title('Wordclouds for open questions')
		df=data[[i for i in data.columns if 'text' in i]].copy()
		#st.write(df)
		feature=st.sidebar.selectbox('Select the question for which you would like to visualize wordclouds of answers',[questions[i] for i in df.columns])	
		var=[i for i in questions if questions[i]==feature][0]
		
		col1, col3 = st.columns([6,3])
		col1.title('Wordcloud from question:')
		col1.title(feature)
				
		x, y = np.ogrid[:300, :300]
		mask = ((x - 150)) ** 2 + ((y - 150)/1.4) ** 2 > 130 ** 2
		mask = 255 * mask.astype(int)
		corpus=' '.join(df[var].apply(lambda x:'' if x=='0' else x))
		corpus=re.sub('[^A-Za-z ]',' ', corpus)
		corpus=re.sub('\s+',' ', corpus)
		corpus=corpus.lower()
		
		col3.title('')
		col3.title('')
		col3.title('')
		sw=col3.multiselect('Select words you would like to remove from the wordcloud', [i[0] for i in Counter(corpus.split(' ')).most_common()[:20] if i[0] not in STOPWORDS])
		
		if corpus==' ':
    			corpus='No_response'
		else:
			corpus=' '.join([i for i in corpus.split(' ') if i not in sw])
		
		wc = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
		
		wc.generate(corpus)
		
		col1.image(wc.to_array(),width=400)	
		
		if col1.checkbox('Would you like to filter Wordcloud according to other questions'):
			
			st.markdown("""---""")
			
			feature2=st.selectbox('Select one question to filter the wordcloud (Select one of the last ones for checking some new tools)',[questions[i] for i in data.columns if \
			i!='FCS Score' and (i in continues or len(data[i].unique())<=8)])
			var2=[i for i in questions if questions[i]==feature2][0]
			
			if var2 in continues:
				threshold=st.slider('Select the threshold', min_value=data[var2].fillna(0).min(),max_value=data[var2].fillna(0).max())
				subcol1,subcol2=st.columns([2,2])	
				
				corpus1=' '.join(data[data[var2]<threshold][var].apply(lambda x:'' if x=='0' else x))
				corpus1=re.sub('[^A-Za-z ]',' ', corpus1)
				corpus1=re.sub('\s+',' ', corpus1)
				corpus1=corpus1.lower()
				if corpus1==' 'or corpus1=='':
    					corpus1='No_response'
				else:
					corpus1=' '.join([i for i in corpus.split(' ') if i not in sw])
				wc1 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
				wc1.generate(corpus1)
				corpus2=' '.join(data[data[var2]>=threshold][var].apply(lambda x:'' if x=='0' else x))
				corpus2=re.sub('[^A-Za-z ]',' ', corpus2)
				corpus2=re.sub('\s+',' ', corpus2)
				corpus2=corpus2.lower()
				if corpus2==' ' or corpus2=='':
    					corpus2='No_response'
				else:
					corpus2=' '.join([i for i in corpus.split(' ') if i not in sw])
				wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
				wc2.generate(corpus2)
				subcol1.write('Response under the threshold')
				subcol1.image(wc1.to_array(),width=400)
				subcol2.write('Response over the threshold')
				subcol2.image(wc2.to_array(),width=400)
			else:
				subcol1,subcol2=st.columns([2,2])
				L=data[var2].unique()
				
				corpus1=corpus2=corpus3=corpus4=corpus5=corpus6=corpus7=corpus8=''
				Corpuses=[corpus1,corpus2,corpus3,corpus4,corpus5,corpus6,corpus7,corpus8]
				
				
				for i in range(len(L)):
					Corpuses[i]=' '.join(data[data[var2]==L[i]][var].apply(lambda x:'' if x=='0' else x))
					Corpuses[i]=re.sub('[^A-Za-z ]',' ', Corpuses[i])
					Corpuses[i]=re.sub('\s+',' ', Corpuses[i])
					Corpuses[i]=Corpuses[i].lower()
					if Corpuses[i]==' ':
    						Corpuses[i]='No_response'
					else:
						Corpuses[i]=' '.join([i for i in Corpuses[i].split(' ') if i not in sw])
					wc2 = WordCloud(background_color="#0E1117", repeat=False, mask=mask)
					wc2.generate(Corpuses[i])
					if i%2==0:
						subcol1.write('Response : '+str(L[i])+' '+str(len(data[data[var2]==L[i]]))+' '+'repondent')
						subcol1.image(wc2.to_array(),width=400)
					else:
						subcol2.write('Response : '+str(L[i])+' '+str(len(data[data[var2]==L[i]]))+' '+'repondent')
						subcol2.image(wc2.to_array(),width=400)
			
	elif topic=='Display Sankey Graphs':
	
		title2.title('Visuals for questions related to cultures (questions C3 to C17)')
		st.title('')
				
			
		sankey=[i for i in data.columns if i[0]=='C' and 'C1_' not in i and 'C2_' not in i and i!='Clan']
		sankeyseeds=sankey[:65]
		sank=data[sankeyseeds]
		bean=sank[[i for i in sank.columns if 'Bean' in i]].copy()
		sesame=sank[[i for i in sank.columns if 'Sesame' in i]].copy()
		cowpea=sank[[i for i in sank.columns if 'Cowpea' in i]].copy()
		maize=sank[[i for i in sank.columns if 'Maize' in i]].copy()
		other=sank[[i for i in sank.columns if 'Other' in i]].copy()
		colonnes=['Seeds Planted','Type of seeds','Origin of seeds','Area cultivated','Did you have enough seed',\
          'Did you face pest attack','Area affected','Have you done pest management','Origin of fertilizer',\
          'Fertilizer from Wardi','Applied good practices','Used irrigation','Area irrigated']
		for i in [bean,sesame,cowpea,maize,other]:
    			i.columns=colonnes
		bean=bean[bean['Seeds Planted']=='Yes']
		sesame=sesame[sesame['Seeds Planted']=='Yes']
		cowpea=cowpea[cowpea['Seeds Planted']=='Yes']
		maize=maize[maize['Seeds Planted']=='Yes']
		other=other[other['Seeds Planted']=='Yes']
		
		bean['Seeds Planted']=bean['Seeds Planted'].apply(lambda x: 'Beans')
		sesame['Seeds Planted']=sesame['Seeds Planted'].apply(lambda x: 'Sesame')
		cowpea['Seeds Planted']=cowpea['Seeds Planted'].apply(lambda x: 'Cowpeas')
		maize['Seeds Planted']=maize['Seeds Planted'].apply(lambda x: 'Maize')
		other['Seeds Planted']=other['Seeds Planted'].apply(lambda x: 'Other')
		
		sank=pd.DataFrame(columns=colonnes)
		for i in [bean,sesame,cowpea,maize,other]:
		    sank=sank.append(i)
		sank['ones']=np.ones(len(sank))
		
		
		
		
		st.title('Some examples')
		
		st.markdown("""---""")
		st.write('Seeds planted - Origin of Seeds - Type of Seeds - Area Cultivated - Did you have enough seeds?')
		fig=sankey_graph(sank,['Seeds Planted','Origin of seeds','Type of seeds','Area cultivated','Did you have enough seed'],height=600,width=1500)
		fig.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.plotly_chart(fig,use_container_width=True)
		
		st.markdown("""---""")
		st.write('Origin of fertilizer - Did you face pest attack - Applied good practices - Seeds Planted')
		fig1=sankey_graph(sank,['Origin of fertilizer','Did you face pest attack','Applied good practices','Seeds Planted'],height=600,width=1500)
		fig1.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.plotly_chart(fig1,use_container_width=True)
		
		st.markdown("""---""")
		st.write('Area Cultivated - Type of Seeds - Did you face pest attack - Area affected')
		fig2=sankey_graph(sank,['Area cultivated','Type of seeds','Did you face pest attack','Area affected'],height=600,width=1500)
		fig2.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
		
		st.plotly_chart(fig2,use_container_width=True)
		
		if st.checkbox('Design my own Sankey Graph'):
			
			st.markdown("""---""")
			feats=st.multiselect('Select features you want to see in the order you want them to appear', colonnes)
			
			if len(feats)>=2:
				st.write(' - '.join(feats))
				fig3=sankey_graph(sank,feats,height=600,width=1500)
				fig3.update_layout(plot_bgcolor='black', paper_bgcolor='grey', width=1500)
				st.plotly_chart(fig3,use_container_width=True)
		
		
		
			
	
	
	else:
		st.title('\t WARDI \t July 2021')	


    
 
if __name__== '__main__':
    main()




    
