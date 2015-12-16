package com.tianyichen.cost;

import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.utils.Matrix;
import com.tianyichen.utils.Scalar;

public class CrossEntropyCost {
	
	public CrossEntropyCost(){}
	
	public static RealMatrix calLoss(RealMatrix predict,RealMatrix real){		
		
		RealMatrix crossLossM=Scalar.crossLoss(predict, real);

		return crossLossM;
	}
	
	
	public static RealMatrix calDelta(RealMatrix predict, RealMatrix real){
		return predict.add(real.scalarMultiply(-1));
	}
	
	

}
