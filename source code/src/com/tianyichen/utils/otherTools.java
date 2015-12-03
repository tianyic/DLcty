package com.tianyichen.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.linear.RealMatrix;

public class otherTools {
	
	public static RealMatrix stochasticSubmatrix(RealMatrix data,int batch_size, Random rng){
		// assume all data has the size number_samples by number_features
		int num_samples=data.getRowDimension();
		int num_features=data.getColumnDimension();
		int batch_num=num_samples/batch_size+1;
		
		// randomly generate a batch index
		int batch_index=rng.nextInt(batch_num);
		List<Integer> rowIndex_tmp=new ArrayList<Integer>();
		
		for(int i=0;i<batch_size;i++){
			if(batch_size*batch_index+i>=num_samples){
				break;
			}else{
				rowIndex_tmp.add(batch_size*batch_index+i);
			}			
		}
		int[] rowIndex=TypeConvert.ArrayTointv(rowIndex_tmp);
		
		//System.out.println(rowIndex_tmp);
		int[] columnIndex=new int[num_features];
		for(int j=0;j<num_features;j++){
			columnIndex[j]=j;
		}
		
		
		//System.out.println(batch_index);
		
		//return null;
		return data.getSubMatrix(rowIndex, columnIndex);
		
	}
	
	public static List<RealMatrix> stochasticSubmatrixLabel(RealMatrix X,RealMatrix Y,int batch_size, Random rng){
		// assume all data has the size number_samples by number_features
		int num_samples=X.getRowDimension();
		int num_features=X.getColumnDimension();
		int batch_num=num_samples/batch_size+1;
		
		// randomly generate a batch index
		int batch_index=rng.nextInt(batch_num);
		List<Integer> rowIndex_tmp=new ArrayList<Integer>();
		
		for(int i=0;i<batch_size;i++){
			if(batch_size*batch_index+i>=num_samples){
				break;
			}else{
				rowIndex_tmp.add(batch_size*batch_index+i);
			}			
		}
		int[] rowIndex=TypeConvert.ArrayTointv(rowIndex_tmp);
		
		//System.out.println(rowIndex_tmp);
		int[] columnIndex=new int[num_features];
		for(int j=0;j<num_features;j++){
			columnIndex[j]=j;
		}
		
		int[] conlumnIndexY={0};
		int[] rowIndexY=rowIndex;
		
		
		//System.out.println(batch_index);
		
		//return null;
		List<RealMatrix> list=new ArrayList<RealMatrix>();
		list.add(X.getSubMatrix(rowIndex, columnIndex));
		list.add(Y.getSubMatrix(rowIndexY, conlumnIndexY));
		return list;
		
	}

    public static int argmax (double [] elems){
    	
     int bestIdx = -1;
     double max = Double.NEGATIVE_INFINITY;
     for (int i = 0; i < elems.length; i++) {
       double elem = elems[i];
       if (elem > max) {
         max = elem;
         bestIdx = i;
       }
     }
     return bestIdx;
   }
}
