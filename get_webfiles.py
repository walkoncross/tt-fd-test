#!/usr/bin/env python
#coding:utf-8
import os
import os.path as osp
#import subprocess

def wget_files(filelist,outpath):
    '''Download data'''
    
    print 'filelist is: ', filelist
    print 'output path is: ', outpath

    if not osp.exists(filelist):
        print 'Cannot find filelist: ', filelist
        return

    f = open(filelist,'r')
    log = open(osp.splitext(filelist)[0] + '_log.txt','w')
    
    
    if not osp.exists(outpath):
        os.mkdir(outpath)
        
    os.chdir(outpath)
    print(os.getcwd())
    for i in f:
        try:
            webfile = str(i.strip())
            basename = osp.basename(webfile)
            if '.' not in basename[-5:]:
				basename = basename+'.jpg'
    
            cmd = 'wget -O '+ basename  + ' ' + webfile 
            print(cmd)            
            if not osp.exists(outpath + basename):
                #status = subprocess.call(cmd)
                status = os.system(cmd)
                if status !=0:
                    print('Failed!!!')
                    log.write('\nFailed: wget ' + webfile)
                    continue
                
                log.write('\nSuccess: wget' + webfile)
                log.flush()
        except Exception, e:
                print 'Failed, got Exception: ', e
                print webfile+'****ENDL'
                log.write('\nFailed with Exception:' + webfile)
                continue   
        
    f.close()
    log.close()

if __name__  =='__main__':
    outpath = './test-images-100'
    file_list = r'./tantan-fd.txt'
    wget_files(file_list,outpath)
    print('END')
