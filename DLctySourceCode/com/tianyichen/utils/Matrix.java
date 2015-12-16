package com.tianyichen.utils;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.summary.Sum;

public class Matrix {
	
	public static RealMatrix mean(RealMatrix matrix,int option){
		// option ==1 return row mean
		// option ==3 return row column mean
		if(option==1){
			double[] rmean=new double[matrix.getRowDimension()];
			for(int i=0;i<matrix.getRowDimension();i++){
				//System.out.println(i);
				//System.out.println(matrix.getColumnDimension());
				double[] row=matrix.getRow(i);
				//System.out.println(row.length);
				//System.out.println(i+"	"+sum(row));
				rmean[i]=sum(row)/(double) matrix.getColumnDimension();
			}
			RealMatrix rowmean=MatrixUtils.createColumnRealMatrix(rmean);
			return rowmean;
		}
		return null;
		
	}
	
	public static double meanMatrix(RealMatrix matrix){

		double mean=0.0;
		double m=(double)matrix.getRowDimension();
		double n=(double)matrix.getColumnDimension();
		for(int i=0;i<matrix.getRowDimension();i++){
			for(int j=0;j<matrix.getColumnDimension();j++){
				mean+=matrix.getEntry(i, j)/(m*n);
			}
		}
		return mean;

	}
	
	public static double sum(double[] vector){
		
		double sum=0.0;
		for(int i=0;i<vector.length;i++){
			sum+=vector[i];
		}
		return sum;
	}

	public static RealMatrix addV(RealMatrix matrix, RealMatrix conlumn){
		RealMatrix m=MatrixUtils.createRealMatrix(matrix.getRowDimension(),matrix.getColumnDimension());
		for(int i=0;i<m.getColumnDimension();i++){
			m.setColumn(i, matrix.getColumnMatrix(i).add(conlumn).getColumn(0));
		}
		return m;
	}
}
