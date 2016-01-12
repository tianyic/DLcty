package com.tianyichen.UI;

import java.util.Map;

import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.MLP.MLP;
import com.tianyichen.MLP.cMLP;
import com.tianyichen.dao.DataStream;

public class UImlp {

	public static void main(String[] args){
		
		String inputX="GSE4226_4227_X.csv";
		String inputY="GSE4226_4227_Y.csv";
		String inputtestX="GSE4229_X.csv";
		String inputtestY="GSE4229_Y.csv";
		
		Map<String, RealMatrix> infoMap=DataStream.loadData(inputX, inputY);
		
		RealMatrix X=infoMap.get("X");
		RealMatrix Y=infoMap.get("Y");
		
		Map<String, RealMatrix> infoTest=DataStream.loadData(inputtestX, inputtestY);
		RealMatrix testX=infoTest.get("X");
		RealMatrix testY=infoTest.get("Y");
		
		int num_features=X.getColumnDimension();
		int num_samples=X.getRowDimension();
		int training_epoches=1000;
		int pretrainingepoch=10;
		int batch_size=23;
		
		double lr=0.3; // learning rate of MLP
		double alpha=0.1; // learning rate of DBN pretraining
		int K=10;
		
		int[] hiddensizes={140,100,100}; // hidden layer sizes.
		cMLP cmlp=new cMLP(X,num_features,hiddensizes,2,true,lr,batch_size,pretrainingepoch,alpha,K);
		cmlp.train(X, Y, batch_size, training_epoches);
		
		System.out.println("=================");
		System.out.println("start to predict labels of test dataset:");
		RealMatrix test_Y=cmlp.convertLabelToVector(testY,2);
		cmlp.predict(testX, test_Y);
		
		
	}
}
