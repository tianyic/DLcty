package com.tianyichen.utils;

public class printTool {
	
	public static void matrix(double[][] m){
		
		System.out.println("Matrix is:");
		for(int i=0;i<m.length;i++){
			for(int j=0;j<m[0].length;j++){
				System.out.print(m[i][j]+"  ");
			}
			System.out.println();
		}
	}
	
	public static void vector(double[] v){
		
		System.out.println("vector is:");
		for(int i=0;i<v.length;i++){
			System.out.print(v[i]+"  ");
		}
		System.out.println();
	}
	
	

}
