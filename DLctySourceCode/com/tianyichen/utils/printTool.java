package com.tianyichen.utils;

import org.apache.commons.math3.linear.RealMatrix;

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
	
	public static void shape(RealMatrix matrix){
		System.out.println("matrix shape:");
		System.out.println("nrows:"+matrix.getRowDimension()+" , ncol:"+matrix.getColumnDimension());
	}

}
