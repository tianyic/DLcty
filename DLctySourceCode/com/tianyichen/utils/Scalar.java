package com.tianyichen.utils;

import java.util.Random;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class Scalar {
	
	
	// exp elementwise
	public static RealMatrix exp(RealMatrix matrix){
		RealMatrix m=MatrixUtils.createRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
		
		for(int i=0;i<matrix.getRowDimension();i++){
			for(int j=0;j<matrix.getColumnDimension();j++){
				m.setEntry(i, j, Math.exp(matrix.getEntry(i, j)));
			}
		}
		return m;
		
	}
	
	// log elementwise
	public static RealMatrix log(RealMatrix matrix){
		RealMatrix m=MatrixUtils.createRealMatrix(matrix.getRowDimension(), matrix.getColumnDimension());
		for(int i=0;i<matrix.getRowDimension();i++){
			for(int j=0;j<matrix.getColumnDimension();j++){
				m.setEntry(i, j, Math.log(matrix.getEntry(i, j)));
			}
		}
		return m;
	}
	
	// multiply two matrices by element
	public static RealMatrix multiply(RealMatrix A,RealMatrix B){
		RealMatrix m=MatrixUtils.createRealMatrix(A.getRowDimension(), A.getColumnDimension());
		for(int i=0;i<A.getRowDimension();i++){
			for(int j=0;j<A.getColumnDimension();j++){
				m.setEntry(i, j,A.getEntry(i, j)*B.getEntry(i, j));
			}
		}
		return m;
	}
	
	
	// sigmoid function elementwise
	public static RealMatrix sigmoid(RealMatrix sigmoid_activation){
		
		RealMatrix m=MatrixUtils.createRealMatrix(sigmoid_activation.getRowDimension(), sigmoid_activation.getColumnDimension());
		for(int i=0;i<sigmoid_activation.getRowDimension();i++){
			for(int j=0;j<sigmoid_activation.getColumnDimension();j++){
				m.setEntry(i, j, 1.0/(1.0+Math.exp(-sigmoid_activation.getEntry(i, j))));
			}
		}
		return m;
	}
	
	public static RealMatrix sigmoidprime(RealMatrix sigmoid_activation){
		
		RealMatrix m=MatrixUtils.createRealMatrix(sigmoid_activation.getRowDimension(), sigmoid_activation.getColumnDimension());
		m=multiply(sigmoid(sigmoid_activation),sigmoid(sigmoid_activation).scalarMultiply(-1).scalarAdd(1));
		return m;
	}	
	
	// sample from bernulli elementwise
	public static RealMatrix sampleBio(RealMatrix p_vector){
		RealMatrix sample_vector=MatrixUtils.createRealMatrix(p_vector.getRowDimension(), p_vector.getColumnDimension());
		Random rm=new Random();
		for(int i=0;i<p_vector.getRowDimension();i++){
			for(int j=0;j<p_vector.getColumnDimension();j++){
				double p=rm.nextDouble();
				if(p<=p_vector.getEntry(i, j)){
					sample_vector.setEntry(i, j, 1);
				}else{
					sample_vector.setEntry(i, j, 0);
				}
			}
		}
		return sample_vector;
		
	}
	
	
	// cross entropy elementwise
	public static RealMatrix crossLoss(RealMatrix newV,RealMatrix oldV){
		
		RealMatrix oneminusoldV=oldV.scalarMultiply(-1).scalarAdd(1);
		RealMatrix oneminusnewV=newV.scalarMultiply(-1).scalarAdd(1+10e-08);
		RealMatrix crossLossM=multiply(oldV,log(newV.scalarAdd(10e-08))).add(multiply(oneminusoldV,log(oneminusnewV)));
		return crossLossM;
	}
	
}
