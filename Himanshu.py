{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('hello world')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "ldamodel.show_topic(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "ldamodel.show_topic(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "import os\n",
    "import re,glob\n",
    "import datefinder\n",
    "print(os.listdir(\"D:/Kaggle\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "job_path = \"D:/Kaggle/CityofLA/Job Bulletins/\"\n",
    "job_files = os.listdir(job_path)\n",
    "print(\"No of files in Job Bulletins Folder \",len(job_files))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "from wand.image import Image as Img\n",
    "pdf = 'D:/Kaggle/CityofLA/Additional data/PDFs/2014/April 2014/040414/PORT POLICE SERGEANT 3222.pdf\n",
    "print('hello world')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from wand.image import Image as Img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from wand.image import Image as Img\n",
    "pdf = 'D:/Kaggle/CityofLA/Additional data/PDFs/2014/April 2014/040414/PORT POLICE SERGEANT 3222.pdf\n",
    "print('hello world')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "pdf = 'D:/Kaggle/CityofLA/Additional data/PDFs/2014/April 2014/040414/PORT POLICE SERGEANT 3222.pdf'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('hello world')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "'god smiled when he made daughter because he knew he had created love and happiness for every mom and papa'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "import datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "files=[dir for dir in os.walk(\"D:/Kaggle/CityofLA\")]\n",
    "for file in files:\n",
    "    print(os.listdir(file[0]))\n",
    "    print(\"\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "bulletins=os.listdir(\"D:/Kaggle/CityofLA/Job Bulletins/\")\n",
    "additional=os.listdir(\"D:/Kaggle/CityofLA/Additional data/\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "os.chdir(\"D:/Kaggle/CityofLA/Additional data\")\n",
    "job_title = pd.read_csv('job_titles.csv')\n",
    "kaggle_data = pd.read_csv('kaggle_data_dictionary.csv')\n",
    "sample_job = pd.read_csv('sample job class export template.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_job.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_job[sample_job['Field Name']=='SCHOOL_TYPE']['Description']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "bulletins=os.listdir(\"D:/Kaggle/CityofLA/Job Bulletins/\")\n",
    "additional=os.listdir(\"D:/Kaggle/CityofLA/Additional data/\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "os.chdir(\"D:/Kaggle/CityofLA/Additional data\")\n",
    "job_title = pd.read_csv('job_titles.csv')\n",
    "kaggle_data = pd.read_csv('kaggle_data_dictionary.csv')\n",
    "sample_job = pd.read_csv('sample job class export template.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_job.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_job[sample_job['Field Name']=='SCHOOL_TYPE']['Description']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_headings(bulletin):       \n",
    "    \n",
    "    \"\"\"\"function to get the headings from text file\n",
    "        takes a single argument\n",
    "        1.takes single argument list of bulletin files\"\"\"\n",
    "    \n",
    "    with open(\"D:/Kaggle/CityofLA/Job Bulletins/\"+bulletins[bulletin]) as f:    ##reading text files \n",
    "        data=f.read().replace('\\t','').split('\\n')\n",
    "        data=[head for head in data if head.isupper()]\n",
    "        return data\n",
    "        \n",
    "def clean_text(bulletin):      \n",
    "    \n",
    "    \n",
    "    \"\"\"function to do basic data cleaning\n",
    "        takes a single argument\n",
    "        1.takes single argument list of bulletin files\"\"\"\n",
    "                                            \n",
    "    \n",
    "    with open(\"D:/Kaggle/CityofLA/Job Bulletins/\"+bulletins[bulletin]) as f:\n",
    "        data=f.read().replace('\\t','').replace('\\n','')\n",
    "        return data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "def to_dataframe(num,df):\n",
    "    \"\"\"\"function to extract features from job bulletin text files and convert to\n",
    "    pandas dataframe.\n",
    "    function take two arguments \n",
    "                        1.the number of files to be read\n",
    "                        2.dataframe object                                      \"\"\"\n",
    "    \n",
    "\n",
    "    \n",
    "    opendate=re.compile(r'(Open [D,d]ate:)(\\s+)(\\d\\d-\\d\\d-\\d\\d)')       #match open date\n",
    "    \n",
    "    salary=re.compile(r'\\$(\\d+,\\d+)((\\s(to|and)\\s)(\\$\\d+,\\d+))?')       #match salary\n",
    "    \n",
    "    requirements=re.compile(r'(REQUIREMENTS?/\\s?MINIMUM QUALIFICATIONS?)(.*)(PROCESS NOTE)')      #match requirements\n",
    "    \n",
    "    for no in range(0,num):\n",
    "        with open(\"D:/Kaggle/CityofLA/Job Bulletins/\"+bulletins[no],encoding=\"ISO-8859-1\") as f:         #reading files \n",
    "                try:\n",
    "                    file=f.read().replace('\\t','')\n",
    "                    data=file.replace('\\n','')\n",
    "                    headings=[heading for heading in file.split('\\n') if heading.isupper()]             ##getting heading from job bulletin\n",
    "\n",
    "                    sal=re.search(salary,data)\n",
    "                    date=datetime.datetime.strptime(re.search(opendate,data).group(3),'%m-%d-%y')\n",
    "                    try:\n",
    "                        req=re.search(requirements,data).group(2)\n",
    "                    except Exception as e:\n",
    "                        req=re.search('(.*)NOTES?',re.findall(r'(REQUIREMENTS?)(.*)(NOTES?)',\n",
    "                                                              data)[0][1][:1200]).group(1)\n",
    "                    \n",
    "                    duties=re.search(r'(DUTIES)(.*)(REQ[A-Z])',data).group(2)\n",
    "                    try:\n",
    "                        enddate=re.search(\n",
    "                                r'(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\\s(\\d{1,2},\\s\\d{4})'\n",
    "                                ,data).group()\n",
    "                    except Exception as e:\n",
    "                        enddate=np.nan\n",
    "                    \n",
    "                    selection= [z[0] for z in re.findall('([A-Z][a-z]+)((\\s\\.\\s)+)',data)]     ##match selection criteria\n",
    "                    \n",
    "                    df=df.append({'File Name':bulletins[no],'Position':headings[0].lower(),'salary_start':sal.group(1),\n",
    "                               'salary_end':sal.group(5),\"opendate\":date,\"requirements\":req,'duties':duties,\n",
    "                                'deadline':enddate,'selection':selection},ignore_index=True)\n",
    "                    \n",
    "                    \n",
    "                    reg=re.compile(r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)\\s(years?)\\s(of\\sfull(-|\\s)time)')\n",
    "                    df['EXPERIENCE_LENGTH']=df['requirements'].apply(lambda x :  re.search(reg,x).group(1) if re.search(reg,x) is not None  else np.nan)\n",
    "                    df['FULL_TIME_PART_TIME']=df['EXPERIENCE_LENGTH'].apply(lambda x:  'FULL_TIME' if x is not np.nan else np.nan )\n",
    "                    \n",
    "                    reg=re.compile(r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)(\\s|-)(years?)\\s(college)')\n",
    "                    df['EDUCATION_YEARS']=df['requirements'].apply(lambda x :  re.search(reg,x).group(1) if re.search(reg,x) is not None  else np.nan)\n",
    "                    df['SCHOOL_TYPE']=df['EDUCATION_YEARS'].apply(lambda x : 'College or University' if x is not np.nan else np.nan)\n",
    "                    \n",
    "                except Exception as e:\n",
    "                    print('umatched sequence')        \n",
    "           \n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.DataFrame(columns=['File Name','Position','salary_start','salary_end','opendate','requirements','duties','deadline'])\n",
    "df=to_dataframe(683,df)\n",
    "df.to_csv('job class output.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dictionary=pd.DataFrame({'Field Name':['File Name','Position','salary_start','salary_end','opendate',\n",
    "                                            'requirements','duties','deadline','selection','EXPERIENCE_LENGTH','FULL_TIME_PART_TIME','EDUCATION_YEARS','SCHOOL_TYPE'],\n",
    "                             })\n",
    "\n",
    "data_dictionary['Description']=['The file name of the job bulletin from which each record came','The title of the particular class (e.g., Systems Analyst, Carpenter)',\n",
    "                              'The overall salary start','The overall maximum salary','The date the job bulletin opened','Overall requirement that has to be filled',\n",
    "                              'A summary of what someone does in the particular job\\n','The date the job bulletin closed','list of selection criterias','Years required in a particular job class or external role.',\n",
    "                              'Whether the required experience is full-time, part','Years required in a particular education program',\n",
    "                               'School Type: School type required (e.g. college or university, high school)']\n",
    "\n",
    "data_dictionary['Data Type']=['string']*13\n",
    "\n",
    "data_dictionary['Accepts Null Values?']=['Yes']*13"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dictionary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dictionary.to_csv('data dictionary.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dictionary.to_csv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\"\n",
    "    convert salary to proper  form \n",
    "    by removing '$' and ',' symbols.\n",
    "                                    \"\"\"\n",
    "\n",
    "df['salary_start']=[int(sal.split(',')[0]+sal.split(',')[1] ) for sal in df['salary_start']]   \n",
    "df['salary_end']=[sal.replace('$','')  if sal!= None else 0 for sal in df['salary_end']  ]\n",
    "df['salary_end']=[int(sal.split(',')[0]+sal.split(',')[1] ) if type(sal)!=int else 0 for sal in df['salary_end']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "plt.figure(figsize=(7,5))\n",
    "sns.distplot(df['salary_start'])\n",
    "plt.title('salary distribution')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "''''calculating salary start - salary end '''\n",
    "\n",
    "df['salary_diff']=abs(df['salary_start']-df['salary_end'])\n",
    "\n",
    "ranges=df[['Position','salary_diff']].sort_values(by='salary_diff',ascending=False)[:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "ranges"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "experience=df['EXPERIENCE_LENGTH'].value_counts().reset_index()\n",
    "experience['index']=experience['index'].apply(lambda x : x.lower())\n",
    "experience=experience.groupby('index',as_index=False).agg('sum')\n",
    "labels=experience['index']\n",
    "sizes=experience['EXPERIENCE_LENGTH']\n",
    "plt.figure(figsize=(5,7))\n",
    "plt.pie(sizes,explode=(0, 0.1, 0, 0,0,0,0),labels=labels)\n",
    "plt.gca().axis('equal')\n",
    "plt.title('Experience value count')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "x1=df['SCHOOL_TYPE'].value_counts()[0]\n",
    "x2=df['FULL_TIME_PART_TIME'].value_counts()[0]\n",
    "plt.figure(figsize=(10,10))\n",
    "plt.bar(height=[x1,x2],x=['College Degree','Experience'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "''''Extracting month out of opendate timestamp object and counting\n",
    "    the number of each occurence of each months using count_values() '''\n",
    "\n",
    "\n",
    "plt.figure(figsize=(11,5))\n",
    "df['open_month']=[z.month for z in df['opendate']]\n",
    "count=df['open_month'].value_counts(sort=False)\n",
    "sns.barplot(y=count.values,x=count.index,palette='rocket')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "''''Extracting month out of opendate timestamp object and counting\n",
    "    the number of each occurence of each months using count_values() '''\n",
    "\n",
    "\n",
    "plt.figure(figsize=(11,5))\n",
    "df['open_month']=[z.month for z in df['opendate']]\n",
    "count=df['open_month'].value_counts(sort=False)\n",
    "sns.barplot(y=count.values,x=count.index,palette='rocket')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''Extracting weekday out of opendate timestamp object and counting\n",
    "    the number of each occurence of each weekday using count_values() '''\n",
    "\n",
    "\n",
    "plt.figure(figsize=(7,5))\n",
    "\n",
    "df['open_day']=[z.weekday() for z in df['opendate']]\n",
    "count=df['open_day'].value_counts(sort=False)\n",
    "sns.barplot(y=count.values,x=count.index,palette='rocket')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "req=' '.join(text for text in df['requirements'])\n",
    "req"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def show_wordcloud(data, title = None):\n",
    "    \n",
    "    \n",
    "    '''funtion to produce and display wordcloud\n",
    "        taken 2 arguments\n",
    "        1.data to produce wordcloud\n",
    "        2.title of wordcloud'''\n",
    "    \n",
    "    \n",
    "    wordcloud = WordCloud(\n",
    "        background_color='yellow',\n",
    "        stopwords=set(STOPWORDS),\n",
    "        max_words=250,\n",
    "        max_font_size=40, \n",
    "        scale=3,\n",
    "        random_state=1 # chosen at random by flipping a coin; it was heads\n",
    "    ).generate(str(data))\n",
    "\n",
    "    fig = plt.figure(1, figsize=(12, 12))\n",
    "    plt.axis('off')\n",
    "    if title: \n",
    "        fig.suptitle(title, fontsize=20)\n",
    "        fig.subplots_adjust(top=2.3)\n",
    "\n",
    "    plt.imshow(wordcloud)\n",
    "    plt.show()\n",
    "show_wordcloud(req,'REQUIREMENTS')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "import os\n",
    "import re,glob\n",
    "import datefinder\n",
    "print(os.listdir(\"D:/Kaggle\"))\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "from nltk import word_tokenize\n",
    "from gensim.models import Word2Vec\n",
    "from sklearn.manifold import TSNE\n",
    "from collections import Counter\n",
    "import gensim\n",
    "from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator\n",
    "from nltk.stem import WordNetLemmatizer as lem\n",
    "from nltk import word_tokenize\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "job_path = \"D:/Kaggle/CityofLA/Job Bulletins/\"\n",
    "job_files = os.listdir(job_path)\n",
    "print(\"No of files in Job Bulletins Folder \",len(job_files))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "from wand.image import Image as Img\n",
    "pdf = 'D:/Kaggle/CityofLA/Additional data/PDFs/2014/April 2014/040414/PORT POLICE SERGEANT 3222.pdf\n",
    "print('hello world')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "from wand.image import Image as Img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "pdf = 'D:/Kaggle/CityofLA/Additional data/PDFs/2014/April 2014/040414/PORT POLICE SERGEANT 3222.pdf'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('hello world')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "'god smiled when he made daughter because he knew he had created love and happiness for every mom and papa'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "import datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "files=[dir for dir in os.walk(\"D:/Kaggle/CityofLA\")]\n",
    "for file in files:\n",
    "    print(os.listdir(file[0]))\n",
    "    print(\"\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "bulletins=os.listdir(\"D:/Kaggle/CityofLA/Job Bulletins/\")\n",
    "additional=os.listdir(\"D:/Kaggle/CityofLA/Additional data/\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "os.chdir(\"D:/Kaggle/CityofLA/Additional data\")\n",
    "job_title = pd.read_csv('job_titles.csv')\n",
    "kaggle_data = pd.read_csv('kaggle_data_dictionary.csv')\n",
    "sample_job = pd.read_csv('sample job class export template.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_job.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_job[sample_job['Field Name']=='SCHOOL_TYPE']['Description']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_headings(bulletin):       \n",
    "    \n",
    "    \"\"\"\"function to get the headings from text file\n",
    "        takes a single argument\n",
    "        1.takes single argument list of bulletin files\"\"\"\n",
    "    \n",
    "    with open(\"D:/Kaggle/CityofLA/Job Bulletins/\"+bulletins[bulletin]) as f:    ##reading text files \n",
    "        data=f.read().replace('\\t','').split('\\n')\n",
    "        data=[head for head in data if head.isupper()]\n",
    "        return data\n",
    "        \n",
    "def clean_text(bulletin):      \n",
    "    \n",
    "    \n",
    "    \"\"\"function to do basic data cleaning\n",
    "        takes a single argument\n",
    "        1.takes single argument list of bulletin files\"\"\"\n",
    "                                            \n",
    "    \n",
    "    with open(\"D:/Kaggle/CityofLA/Job Bulletins/\"+bulletins[bulletin]) as f:\n",
    "        data=f.read().replace('\\t','').replace('\\n','')\n",
    "        return data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "def to_dataframe(num,df):\n",
    "    \"\"\"\"function to extract features from job bulletin text files and convert to\n",
    "    pandas dataframe.\n",
    "    function take two arguments \n",
    "                        1.the number of files to be read\n",
    "                        2.dataframe object                                      \"\"\"\n",
    "    \n",
    "\n",
    "    \n",
    "    opendate=re.compile(r'(Open [D,d]ate:)(\\s+)(\\d\\d-\\d\\d-\\d\\d)')       #match open date\n",
    "    \n",
    "    salary=re.compile(r'\\$(\\d+,\\d+)((\\s(to|and)\\s)(\\$\\d+,\\d+))?')       #match salary\n",
    "    \n",
    "    requirements=re.compile(r'(REQUIREMENTS?/\\s?MINIMUM QUALIFICATIONS?)(.*)(PROCESS NOTE)')      #match requirements\n",
    "    \n",
    "    for no in range(0,num):\n",
    "        with open(\"D:/Kaggle/CityofLA/Job Bulletins/\"+bulletins[no],encoding=\"ISO-8859-1\") as f:         #reading files \n",
    "                try:\n",
    "                    file=f.read().replace('\\t','')\n",
    "                    data=file.replace('\\n','')\n",
    "                    headings=[heading for heading in file.split('\\n') if heading.isupper()]             ##getting heading from job bulletin\n",
    "\n",
    "                    sal=re.search(salary,data)\n",
    "                    date=datetime.datetime.strptime(re.search(opendate,data).group(3),'%m-%d-%y')\n",
    "                    try:\n",
    "                        req=re.search(requirements,data).group(2)\n",
    "                    except Exception as e:\n",
    "                        req=re.search('(.*)NOTES?',re.findall(r'(REQUIREMENTS?)(.*)(NOTES?)',\n",
    "                                                              data)[0][1][:1200]).group(1)\n",
    "                    \n",
    "                    duties=re.search(r'(DUTIES)(.*)(REQ[A-Z])',data).group(2)\n",
    "                    try:\n",
    "                        enddate=re.search(\n",
    "                                r'(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\\s(\\d{1,2},\\s\\d{4})'\n",
    "                                ,data).group()\n",
    "                    except Exception as e:\n",
    "                        enddate=np.nan\n",
    "                    \n",
    "                    selection= [z[0] for z in re.findall('([A-Z][a-z]+)((\\s\\.\\s)+)',data)]     ##match selection criteria\n",
    "                    \n",
    "                    df=df.append({'File Name':bulletins[no],'Position':headings[0].lower(),'salary_start':sal.group(1),\n",
    "                               'salary_end':sal.group(5),\"opendate\":date,\"requirements\":req,'duties':duties,\n",
    "                                'deadline':enddate,'selection':selection},ignore_index=True)\n",
    "                    \n",
    "                    \n",
    "                    reg=re.compile(r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)\\s(years?)\\s(of\\sfull(-|\\s)time)')\n",
    "                    df['EXPERIENCE_LENGTH']=df['requirements'].apply(lambda x :  re.search(reg,x).group(1) if re.search(reg,x) is not None  else np.nan)\n",
    "                    df['FULL_TIME_PART_TIME']=df['EXPERIENCE_LENGTH'].apply(lambda x:  'FULL_TIME' if x is not np.nan else np.nan )\n",
    "                    \n",
    "                    reg=re.compile(r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)(\\s|-)(years?)\\s(college)')\n",
    "                    df['EDUCATION_YEARS']=df['requirements'].apply(lambda x :  re.search(reg,x).group(1) if re.search(reg,x) is not None  else np.nan)\n",
    "                    df['SCHOOL_TYPE']=df['EDUCATION_YEARS'].apply(lambda x : 'College or University' if x is not np.nan else np.nan)\n",
    "                    \n",
    "                except Exception as e:\n",
    "                    print('umatched sequence')        \n",
    "           \n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.DataFrame(columns=['File Name','Position','salary_start','salary_end','opendate','requirements','duties','deadline'])\n",
    "df=to_dataframe(683,df)\n",
    "df.to_csv('job class output.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dictionary=pd.DataFrame({'Field Name':['File Name','Position','salary_start','salary_end','opendate',\n",
    "                                            'requirements','duties','deadline','selection','EXPERIENCE_LENGTH','FULL_TIME_PART_TIME','EDUCATION_YEARS','SCHOOL_TYPE'],\n",
    "                             })\n",
    "\n",
    "data_dictionary['Description']=['The file name of the job bulletin from which each record came','The title of the particular class (e.g., Systems Analyst, Carpenter)',\n",
    "                              'The overall salary start','The overall maximum salary','The date the job bulletin opened','Overall requirement that has to be filled',\n",
    "                              'A summary of what someone does in the particular job\\n','The date the job bulletin closed','list of selection criterias','Years required in a particular job class or external role.',\n",
    "                              'Whether the required experience is full-time, part','Years required in a particular education program',\n",
    "                               'School Type: School type required (e.g. college or university, high school)']\n",
    "\n",
    "data_dictionary['Data Type']=['string']*13\n",
    "\n",
    "data_dictionary['Accepts Null Values?']=['Yes']*13"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dictionary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dictionary.to_csv('data dictionary.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dictionary.to_csv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\"\n",
    "    convert salary to proper  form \n",
    "    by removing '$' and ',' symbols.\n",
    "                                    \"\"\"\n",
    "\n",
    "df['salary_start']=[int(sal.split(',')[0]+sal.split(',')[1] ) for sal in df['salary_start']]   \n",
    "df['salary_end']=[sal.replace('$','')  if sal!= None else 0 for sal in df['salary_end']  ]\n",
    "df['salary_end']=[int(sal.split(',')[0]+sal.split(',')[1] ) if type(sal)!=int else 0 for sal in df['salary_end']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "plt.figure(figsize=(7,5))\n",
    "sns.distplot(df['salary_start'])\n",
    "plt.title('salary distribution')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "''''calculating salary start - salary end '''\n",
    "\n",
    "df['salary_diff']=abs(df['salary_start']-df['salary_end'])\n",
    "\n",
    "ranges=df[['Position','salary_diff']].sort_values(by='salary_diff',ascending=False)[:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "ranges"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "experience=df['EXPERIENCE_LENGTH'].value_counts().reset_index()\n",
    "experience['index']=experience['index'].apply(lambda x : x.lower())\n",
    "experience=experience.groupby('index',as_index=False).agg('sum')\n",
    "labels=experience['index']\n",
    "sizes=experience['EXPERIENCE_LENGTH']\n",
    "plt.figure(figsize=(5,7))\n",
    "plt.pie(sizes,explode=(0, 0.1, 0, 0,0,0,0),labels=labels)\n",
    "plt.gca().axis('equal')\n",
    "plt.title('Experience value count')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "x1=df['SCHOOL_TYPE'].value_counts()[0]\n",
    "x2=df['FULL_TIME_PART_TIME'].value_counts()[0]\n",
    "plt.figure(figsize=(10,10))\n",
    "plt.bar(height=[x1,x2],x=['College Degree','Experience'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "''''Extracting month out of opendate timestamp object and counting\n",
    "    the number of each occurence of each months using count_values() '''\n",
    "\n",
    "\n",
    "plt.figure(figsize=(11,5))\n",
    "df['open_month']=[z.month for z in df['opendate']]\n",
    "count=df['open_month'].value_counts(sort=False)\n",
    "sns.barplot(y=count.values,x=count.index,palette='rocket')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''Extracting weekday out of opendate timestamp object and counting\n",
    "    the number of each occurence of each weekday using count_values() '''\n",
    "\n",
    "\n",
    "plt.figure(figsize=(7,5))\n",
    "\n",
    "df['open_day']=[z.weekday() for z in df['opendate']]\n",
    "count=df['open_day'].value_counts(sort=False)\n",
    "sns.barplot(y=count.values,x=count.index,palette='rocket')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "req=' '.join(text for text in df['requirements'])\n",
    "req"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def show_wordcloud(data, title = None):\n",
    "    \n",
    "    \n",
    "    '''funtion to produce and display wordcloud\n",
    "        taken 2 arguments\n",
    "        1.data to produce wordcloud\n",
    "        2.title of wordcloud'''\n",
    "    \n",
    "    \n",
    "    wordcloud = WordCloud(\n",
    "        background_color='yellow',\n",
    "        stopwords=set(STOPWORDS),\n",
    "        max_words=250,\n",
    "        max_font_size=40, \n",
    "        scale=3,\n",
    "        random_state=1 # chosen at random by flipping a coin; it was heads\n",
    "    ).generate(str(data))\n",
    "\n",
    "    fig = plt.figure(1, figsize=(12, 12))\n",
    "    plt.axis('off')\n",
    "    if title: \n",
    "        fig.suptitle(title, fontsize=20)\n",
    "        fig.subplots_adjust(top=2.3)\n",
    "\n",
    "    plt.imshow(wordcloud)\n",
    "    plt.show()\n",
    "show_wordcloud(req,'REQUIREMENTS')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "lem=WordNetLemmatizer()\n",
    "text=[lem.lemmatize(w) for w in word_tokenize(req)]\n",
    "vect=TfidfVectorizer(ngram_range=(1,3),max_features=100)\n",
    "vectorized_data=vect.fit_transform(text)\n",
    "#id_map=dict((v,k) for k,v in vect.vocabulary_.items())\n",
    "vect.vocabulary_.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator\n",
    "from nltk.stem import WordNetLemmatizer as lem\n",
    "from nltk import word_tokenize\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    "from PIL import Image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.stem import WordNetLemmatizer\n",
    "from nltk import word_tokenize\n",
    "from gensim.models import Word2Vec\n",
    "from sklearn.manifold import TSNE\n",
    "from collections import Counter\n",
    "import gensim\n",
    "from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator\n",
    "from nltk.stem import WordNetLemmatizer as lem\n",
    "from nltk import word_tokenize\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_corpus(df,col):\n",
    "    \n",
    "    '''function to build corpus from dataframe'''\n",
    "    lem=WordNetLemmatizer()\n",
    "    corpus= []\n",
    "    for x in df[col]:\n",
    "        \n",
    "        \n",
    "        words=word_tokenize(x)\n",
    "        corpus.append([lem.lemmatize(w) for w in words])\n",
    "    return corpus"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [],
   "source": [
    "corpus=build_corpus(df,'requirements')\n",
    "model = Word2Vec(corpus, size=100, window=20, min_count=30, workers=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk\n",
    "word2vec."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {},
   "outputs": [],
   "source": [
    "nltk.download('punkt')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [],
   "source": [
    "nltk.download('wordnet')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {},
   "outputs": [],
   "source": [
    "def tsne_plot(model,title='None'):\n",
    "    \"Creates and TSNE model and plots it\"\n",
    "    labels = []\n",
    "    tokens = []\n",
    "\n",
    "    for word in model.wv.vocab:\n",
    "        tokens.append(model[word])\n",
    "        labels.append(word)\n",
    "    \n",
    "    tsne_model = TSNE(perplexity=80, n_components=2, init='pca', n_iter=2500, random_state=23)\n",
    "    new_values = tsne_model.fit_transform(tokens)\n",
    "\n",
    "    x = []\n",
    "    y = []\n",
    "    for value in new_values:\n",
    "        x.append(value[0])\n",
    "        y.append(value[1])\n",
    "        \n",
    "    plt.figure(figsize=(15, 15)) \n",
    "    plt.title(title)\n",
    "    for i in range(len(x)):\n",
    "        plt.scatter(x[i],y[i])\n",
    "        plt.annotate(labels[i],\n",
    "                     xy=(x[i], y[i]),\n",
    "                     xytext=(5, 2),\n",
    "                     textcoords='offset points',\n",
    "                     ha='right',\n",
    "                     va='bottom')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [],
   "source": [
    "tsne_plot(model,'Requirements')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [],
   "source": [
    "token=word_tokenize(req)\n",
    "counter=Counter(token)\n",
    "count=[x[0] for x in counter.most_common(40) if len(x[0])>3]\n",
    "print(\"Most common words in Requirement\")\n",
    "print(count)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [],
   "source": [
    "duties= ' '.join(d for d in df['duties'])\n",
    "show_wordcloud(duties,'Duties')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [],
   "source": [
    "lem=WordNetLemmatizer()\n",
    "text=[lem.lemmatize(w) for w in word_tokenize(duties)]\n",
    "vect=TfidfVectorizer(ngram_range=(1,3),max_features=200)\n",
    "vectorized_data=vect.fit_transform(text)\n",
    "#id_map=dict((v,k) for k,v in vect.vocabulary_.items())\n",
    "vect.vocabulary_.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [],
   "source": [
    "token=word_tokenize(duties)\n",
    "counter=Counter(token)\n",
    "count=[x[0] for x in counter.most_common(40) if len(x[0])>3]\n",
    "print(\"Most common words in Duties\")\n",
    "print(count)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [],
   "source": [
    "corpus=build_corpus(df,'duties')\n",
    "model =Word2Vec(corpus, size=100, window=20, min_count=40, workers=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [],
   "source": [
    "tsne_plot(model,'Duties')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [],
   "source": [
    "lem=WordNetLemmatizer()\n",
    "text=[lem.lemmatize(w) for w in word_tokenize(duties)]\n",
    "vect=TfidfVectorizer(ngram_range=(1,3),max_features=200)\n",
    "vectorized_data=vect.fit_transform(text)\n",
    "id2word=dict((v,k) for k,v in vect.vocabulary_.items())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [],
   "source": [
    "corpus=gensim.matutils.Sparse2Corpus(vectorized_data,documents_columns=False)\n",
    "ldamodel = gensim.models.ldamodel.LdaModel(corpus,id2word=id2word,num_topics=8,random_state=34,passes=25,per_word_topics=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [],
   "source": [
    "ldamodel.show_topic(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [],
   "source": [
    "ldamodel.show_topic(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {},
   "outputs": [],
   "source": [
    "data=build_corpus(df,'duties')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(7,7))\n",
    "count=df['selection'].astype(str).value_counts()[:10]\n",
    "sns.barplot(y=count.index,x=count,palette='rocket')\n",
    "plt.gca().set_yticklabels(count.index,rotation='45')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {},
   "outputs": [],
   "source": [
    "def pronoun(data):\n",
    "    \n",
    "    '''function to tokenize data and perform pos_tagging.Returns tokens having \"PRP\" tag'''\n",
    "    \n",
    "    prn=[]\n",
    "    vrb=[]\n",
    "    token=word_tokenize(data)\n",
    "    pos=pos_tag(token)\n",
    "   \n",
    "    vrb=Counter([x[0] for x in pos if x[1]=='PRP'])\n",
    "    \n",
    "    return vrb\n",
    "    \n",
    "\n",
    "\n",
    "req_prn=pronoun(req)\n",
    "duties_prn=pronoun(duties)\n",
    "print('pronouns used in requirement section are')\n",
    "print(req_prn.keys())\n",
    "print('\\npronouns used in duties section are')\n",
    "print(duties_prn.keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "import os\n",
    "import re,glob\n",
    "import datefinder\n",
    "print(os.listdir(\"D:/Kaggle\"))\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "from nltk import word_tokenize\n",
    "from gensim.models import Word2Vec\n",
    "from sklearn.manifold import TSNE\n",
    "from collections import Counter\n",
    "import gensim\n",
    "from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator\n",
    "from nltk.stem import WordNetLemmatizer as lem\n",
    "from nltk import word_tokenize\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from nltk import pos_tag"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [],
   "source": [
    "def pronoun(data):\n",
    "    \n",
    "    '''function to tokenize data and perform pos_tagging.Returns tokens having \"PRP\" tag'''\n",
    "    \n",
    "    prn=[]\n",
    "    vrb=[]\n",
    "    token=word_tokenize(data)\n",
    "    pos=pos_tag(token)\n",
    "   \n",
    "    vrb=Counter([x[0] for x in pos if x[1]=='PRP'])\n",
    "    \n",
    "    return vrb\n",
    "    \n",
    "\n",
    "\n",
    "req_prn=pronoun(req)\n",
    "duties_prn=pronoun(duties)\n",
    "print('pronouns used in requirement section are')\n",
    "print(req_prn.keys())\n",
    "print('\\npronouns used in duties section are')\n",
    "print(duties_prn.keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {},
   "outputs": [],
   "source": [
    "def pronoun(data):\n",
    "    \n",
    "    '''function to tokenize data and perform pos_tagging.Returns tokens having \"PRP\" tag'''\n",
    "    \n",
    "    prn=[]\n",
    "    vrb=[]\n",
    "    token=word_tokenize(data)\n",
    "    pos=nltk.pos_tag(token)\n",
    "   \n",
    "    vrb=Counter([x[0] for x in pos if x[1]=='PRP'])\n",
    "    \n",
    "    return vrb\n",
    "    \n",
    "\n",
    "\n",
    "req_prn=pronoun(req)\n",
    "duties_prn=pronoun(duties)\n",
    "print('pronouns used in requirement section are')\n",
    "print(req_prn.keys())\n",
    "print('\\npronouns used in duties section are')\n",
    "print(duties_prn.keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "metadata": {},
   "outputs": [],
   "source": [
    "word_tokenize(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [],
   "source": [
    "def pronoun(data):\n",
    "    \n",
    "    '''function to tokenize data and perform pos_tagging.Returns tokens having \"PRP\" tag'''\n",
    "    \n",
    "    prn=[]\n",
    "    vrb=[]\n",
    "    token=nltk.word_tokenize(data)\n",
    "    pos=nltk.pos_tag(token)\n",
    "   \n",
    "    vrb=Counter([x[0] for x in pos if x[1]=='PRP'])\n",
    "    \n",
    "    return vrb\n",
    "    \n",
    "\n",
    "\n",
    "req_prn=pronoun(req)\n",
    "duties_prn=pronoun(duties)\n",
    "print('pronouns used in requirement section are')\n",
    "print(req_prn.keys())\n",
    "print('\\npronouns used in duties section are')\n",
    "print(duties_prn.keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk import word_tokenize"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "metadata": {},
   "outputs": [],
   "source": [
    "word_tokenize(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "metadata": {},
   "outputs": [],
   "source": [
    "word_tokenize(req)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "metadata": {},
   "outputs": [],
   "source": [
    "def pronoun(data):\n",
    "    \n",
    "    '''function to tokenize data and perform pos_tagging.Returns tokens having \"PRP\" tag'''\n",
    "    \n",
    "    prn=[]\n",
    "    vrb=[]\n",
    "    token=word_tokenize(data)\n",
    "    pos=pos_tag(token)\n",
    "   \n",
    "    vrb=Counter([x[0] for x in pos if x[1]=='PRP'])\n",
    "    \n",
    "    return vrb\n",
    "    \n",
    "\n",
    "\n",
    "req_prn=pronoun(req)\n",
    "duties_prn=pronoun(duties)\n",
    "print('pronouns used in requirement section are')\n",
    "print(req_prn.keys())\n",
    "print('\\npronouns used in duties section are')\n",
    "print(duties_prn.keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk import word_tokenize\n",
    "from nltk import pos_tag"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "metadata": {},
   "outputs": [],
   "source": [
    "pos_tag(req)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [],
   "source": [
    "nltk.download('averaged_perceptron_tagger')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk import word_tokenize\n",
    "from nltk import pos_tag"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {},
   "outputs": [],
   "source": [
    "pos_tag(req)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {},
   "outputs": [],
   "source": [
    "def pronoun(data):\n",
    "    \n",
    "    '''function to tokenize data and perform pos_tagging.Returns tokens having \"PRP\" tag'''\n",
    "    \n",
    "    prn=[]\n",
    "    vrb=[]\n",
    "    token=word_tokenize(data)\n",
    "    pos=pos_tag(token)\n",
    "   \n",
    "    vrb=Counter([x[0] for x in pos if x[1]=='PRP'])\n",
    "    \n",
    "    return vrb\n",
    "    \n",
    "\n",
    "\n",
    "req_prn=pronoun(req)\n",
    "duties_prn=pronoun(duties)\n",
    "print('pronouns used in requirement section are')\n",
    "print(req_prn.keys())\n",
    "print('\\npronouns used in duties section are')\n",
    "print(duties_prn.keys())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {},
   "outputs": [],
   "source": [
    "names=['senior waterman','policeman']\n",
    "for name in names:\n",
    "    z=re.match(r'\\w+?\\s?\\w+(man|women)$',name)\n",
    "    print(z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "metadata": {},
   "outputs": [],
   "source": [
    "for name in df['Position']:\n",
    "    z=re.match(r'\\w+?\\s?\\w+(man|women)$',name)\n",
    "    if z is not None:\n",
    "        print(z)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "metadata": {},
   "outputs": [],
   "source": [
    "def similar_jobs(job):\n",
    "    \n",
    "    ''' function to find and return jobs with similar job title.take a single argument\n",
    "            - job title\n",
    "            returns\n",
    "                -list of similar jobs '''\n",
    "    \n",
    "    word1=word_tokenize(job)\n",
    "    jobs=[]\n",
    "    for i,name in enumerate(df['Position']):\n",
    "        word2=word_tokenize(name)\n",
    "        distance=jaccard_distance(set(ngrams(word1,n=1)),set(ngrams(word2,n=1)))\n",
    "        if(distance<.55):\n",
    "            jobs.append((name,i))\n",
    "    return jobs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "metadata": {},
   "outputs": [],
   "source": [
    "similar_jobs(df['Position'][10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk import jaccard distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk import jaccard_distance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "metadata": {},
   "outputs": [],
   "source": [
    "def similar_jobs(job):\n",
    "    \n",
    "    ''' function to find and return jobs with similar job title.take a single argument\n",
    "            - job title\n",
    "            returns\n",
    "                -list of similar jobs '''\n",
    "    \n",
    "    word1=word_tokenize(job)\n",
    "    jobs=[]\n",
    "    for i,name in enumerate(df['Position']):\n",
    "        word2=word_tokenize(name)\n",
    "        distance=jaccard_distance(set(ngrams(word1,n=1)),set(ngrams(word2,n=1)))\n",
    "        if(distance<.55):\n",
    "            jobs.append((name,i))\n",
    "    return jobs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {},
   "outputs": [],
   "source": [
    "similar_jobs(df['Position'][10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [],
   "source": [
    "def similar_jobs(job):\n",
    "    \n",
    "    ''' function to find and return jobs with similar job title.take a single argument\n",
    "            - job title\n",
    "            returns\n",
    "                -list of similar jobs '''\n",
    "    \n",
    "    word1=word_tokenize(job)\n",
    "    jobs=[]\n",
    "    for i,name in enumerate(df['Position']):\n",
    "        word2=word_tokenize(name)\n",
    "        distance=jaccard_distance(set(ngrams(word1,n=1)),set(ngrams(word2,n=1)))\n",
    "        if(distance<.55):\n",
    "            jobs.append((name,i))\n",
    "    return jobs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [],
   "source": [
    "similar_jobs(df['Position'][10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk import jaccard_distance\n",
    "from nltk import ngrams"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [],
   "source": [
    "def similar_jobs(job):\n",
    "    \n",
    "    ''' function to find and return jobs with similar job title.take a single argument\n",
    "            - job title\n",
    "            returns\n",
    "                -list of similar jobs '''\n",
    "    \n",
    "    word1=word_tokenize(job)\n",
    "    jobs=[]\n",
    "    for i,name in enumerate(df['Position']):\n",
    "        word2=word_tokenize(name)\n",
    "        distance=jaccard_distance(set(ngrams(word1,n=1)),set(ngrams(word2,n=1)))\n",
    "        if(distance<.55):\n",
    "            jobs.append((name,i))\n",
    "    return jobs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [],
   "source": [
    "similar_jobs(df['Position'][10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [],
   "source": [
    "def similar_req(job):\n",
    "    \n",
    "    ''' function to find and return jobs with similar job title.take a single argument\n",
    "            - job title\n",
    "            returns\n",
    "                -list of similar jobs '''\n",
    "    \n",
    "    word1=word_tokenize(job)\n",
    "    jobs=[]\n",
    "    for i,name in enumerate(df['requirements']):\n",
    "        word2=word_tokenize(name)\n",
    "        distance=jaccard_distance(set(ngrams(word1,n=1)),set(ngrams(word2,n=1)))\n",
    "        if(distance<.5):\n",
    "            jobs.append((name,df.iloc[i]['Position']))\n",
    "    return jobs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [],
   "source": [
    "def similar_req(job):\n",
    "    \n",
    "    ''' function to find and return jobs with similar job title.take a single argument\n",
    "            - job title\n",
    "            returns\n",
    "                -list of similar jobs '''\n",
    "    \n",
    "    word1=word_tokenize(job)\n",
    "    jobs=[]\n",
    "    for i,name in enumerate(df['requirements']):\n",
    "        word2=word_tokenize(name)\n",
    "        distance=jaccard_distance(set(ngrams(word1,n=1)),set(ngrams(word2,n=1)))\n",
    "        if(distance<.5):\n",
    "            jobs.append((name,df.iloc[i]['Position']))\n",
    "    return jobs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [],
   "source": [
    "reading=[]\n",
    "for file in df['File Name']:\n",
    "    text=open(\"../input/cityofla/CityofLA/Job Bulletins/\"+file,'r',encoding=\"ISO-8859-1\").read()\n",
    "    sentence = text.count('.') + text.count('!') + text.count(';') + text.count(':') + text.count('?')\n",
    "    words = len(text.split())\n",
    "    syllable = 0\n",
    "    for word in text.split():\n",
    "        for vowel in ['a','e','i','o','u']:\n",
    "            syllable += word.count(vowel)\n",
    "        for ending in ['es','ed','e']:\n",
    "            if word.endswith(ending):\n",
    "                   syllable -= 1\n",
    "        if word.endswith('le'):\n",
    "            syllable += 1\n",
    "            \n",
    "    G = round((0.39*words)/sentence+ (11.8*syllable)/words-15.59)\n",
    "    reading.append(G)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [],
   "source": [
    "reading=[]\n",
    "for file in df['File Name']:\n",
    "    text=open(\"D:/Kaggle/CityofLA/Job Bulletins/\"+file,'r',encoding=\"ISO-8859-1\").read()\n",
    "    sentence = text.count('.') + text.count('!') + text.count(';') + text.count(':') + text.count('?')\n",
    "    words = len(text.split())\n",
    "    syllable = 0\n",
    "    for word in text.split():\n",
    "        for vowel in ['a','e','i','o','u']:\n",
    "            syllable += word.count(vowel)\n",
    "        for ending in ['es','ed','e']:\n",
    "            if word.endswith(ending):\n",
    "                   syllable -= 1\n",
    "        if word.endswith('le'):\n",
    "            syllable += 1\n",
    "            \n",
    "    G = round((0.39*words)/sentence+ (11.8*syllable)/words-15.59)\n",
    "    reading.append(G)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.hist(reading)\n",
    "plt.xlabel('Flesch Index')\n",
    "plt.title('Flesch index distribution')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "import os\n",
    "import re,glob\n",
    "import datefinder\n",
    "print(os.listdir(\"D:/Kaggle\"))\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "from nltk import word_tokenize\n",
    "from gensim.models import Word2Vec\n",
    "from sklearn.manifold import TSNE\n",
    "from collections import Counter\n",
    "import gensim\n",
    "from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator\n",
    "from nltk.stem import WordNetLemmatizer as lem\n",
    "from nltk import word_tokenize\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from nltk import pos_tag\n",
    "from nltk import jaccard_distance\n",
    "from nltk import ngrams"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [],
   "source": [
    "!jupyter nbconvert --to script config_template.ipynb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [],
   "source": [
    "!jupyter nbconvert --to script config_template.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {},
   "outputs": [],
   "source": [
    " %notebook \"D:/Kaggle/Himanshu.ipynb\"   "
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 2
}
