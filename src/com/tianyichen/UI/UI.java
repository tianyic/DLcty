package com.tianyichen.UI;

import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.RBM.RBM;
import com.tianyichen.dao.DataStream;
import com.tianyichen.utils.otherTools;
import com.tianyichen.utils.printTool;

public class UI {
	
	public static void main(String[] args){
		
		
		
		
		String inputX="german_numer01_X.out";
		String inputY="german_numer01_Y.out";
		//String inputX="ijcnn1_X.out";
		//String inputY="ijcnn1_Y.out";
		Map<String, RealMatrix> infoMap=DataStream.loadData(inputX, inputY);
		
		RealMatrix X=infoMap.get("X");
		RealMatrix Y=infoMap.get("Y");
		
		int num_features=X.getColumnDimension();
		int num_samples=X.getRowDimension();
		num_samples=Y.getRowDimension();
		
		System.out.println("input dataset size:");
		System.out.println("number of features: "+num_features+", number of samples:"+num_samples);
		
		RBM rbm1=new RBM(num_features, num_features/2);
		
		System.out.println("W size:");
		System.out.println("number of rows: "+rbm1.getW().getRowDimension()+", number of columns:"+rbm1.getW().getColumnDimension());
		
		System.out.println("hbias size:");
		System.out.println("number of rows: "+rbm1.getHbias().getRowDimension()+", number of columns:"+rbm1.getHbias().getColumnDimension());

		System.out.println("vbias size:");
		System.out.println("number of rows: "+rbm1.getVbias().getRowDimension()+", number of columns:"+rbm1.getVbias().getColumnDimension());

		int training_epoches=100;
		int batch_size=111;
		int K=1;
		double alpha=0.1;
		rbm1.setK(K);
		rbm1.setAlpha(alpha);
		for(int t=0;t<training_epoches;t++){
			Random randomGenerator = new Random();
			RealMatrix inputData=otherTools.stochasticSubmatrix(X, batch_size, randomGenerator);
			//System.out.println("input data size:");
			//System.out.println("number of rows: "+inputData.getRowDimension()+", number of columns:"+inputData.getColumnDimension());
			//printTool.matrix(inputData.getData());
			rbm1.updateParams(inputData);

			double loss=rbm1.CrossEntropy(inputData);
			System.out.println("loss value:"+loss);
			
		}
		
		//printTool.matrix(rbm1.getW().getData());
	
	}	

}
