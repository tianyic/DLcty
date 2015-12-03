package com.tianyichen.utils;

import java.util.List;

public class TypeConvert {
	
	public static double[] DoubleTodoublev(Double[] array){
		
		double[] v1=new double[array.length];
		for(int i=0;i<array.length;i++){
			v1[i]=array[i];
		}
		return v1;
	}
	
	public static double[][] DoubleTodoublem(Double[][] array){
		double[][] m1=new double[array.length][array[0].length];
		
		for(int i=0;i<array.length;i++){
			for(int j=0;j<array[0].length;j++){
				m1[i][j]=array[i][j];
			}
		}
		return m1;
	}
	
	
	public static double[][] ArrayTodoublem(List<List> array){
		double[][] m1=new double[array.size()][array.get(0).size()];
		
		for(int i=0;i<array.size();i++){
			for(int j=0;j<array.get(0).size();j++){
				m1[i][j]=(double) array.get(i).get(j);
			}
		}
		return m1;
	}
	
	public static double[] ArrayTodoublev(List array){
		double[] v1=new double[array.size()];
		
		for(int i=0;i<array.size();i++){

			v1[i]=(double) array.get(i);

		}
		return v1;
	}
	public static int[] ArrayTointv(List array){
		int[] v1=new int[array.size()];
		
		for(int i=0;i<array.size();i++){

			v1[i]=(int) array.get(i);

		}
		return v1;
	}
}
