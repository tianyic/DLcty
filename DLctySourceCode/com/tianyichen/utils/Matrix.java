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
		}else if(option==2){
			double[] cmean=new double[matrix.getColumnDimension()];
			for(int i=0;i<matrix.getColumnDimension();i++){
				double[] column=matrix.getColumn(i);
				cmean[i]=sum(column)/(double) matrix.getRowDimension();
			}
			RealMatrix colmean=MatrixUtils.createColumnRealMatrix(cmean);
			return colmean;
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
	
	public static double[] colMeans(RealMatrix matrix){
		double[] means=new double[matrix.getColumnDimension()];
		for(int i=0;i<matrix.getColumnDimension();i++){
			means[i]=mean(matrix.getColumn(i));
		}
		
		return means;
	}
	
	public static double[] rowMeans(RealMatrix matrix){
		double[] means=new double[matrix.getRowDimension()];
		
		for(int i=0;i<matrix.getRowDimension();i++){
			means[i]=mean(matrix.getRow(i));
		}
		
		return means;		
	}
	
	public static double mean(double[] array ){
		double mean=0.0;
		for(int i=0;i<array.length;i++){
			mean+=array[i]/(double)array.length;
		}
		return mean;
	}
	
	public static double[] colVariances(RealMatrix matrix){
		double[] var=new double[matrix.getColumnDimension()];
		for(int i=0;i<matrix.getColumnDimension();i++){
			var[i]=variance(matrix.getColumn(i));
		}
		return var;
	}
	
	public static double[] rowVariances(RealMatrix matrix){
		
		double[] var=new double[matrix.getRowDimension()];		
		for(int i=0;i<matrix.getRowDimension();i++){
			var[i]=variance(matrix.getRow(i));
		}
		
		return var;		
	}
	
	public static double variance(double[] array){
		
		double variance=0.0;
		double mean=mean(array);
		
		for(int i=0;i<array.length;i++){
			
			variance+=Math.pow(array[i]-mean, 2)/(double)array.length;
		
		}
		
		return variance;
		
	}
}
