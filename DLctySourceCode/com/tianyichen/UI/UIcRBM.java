package com.tianyichen.UI;

import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.RBM.RBM;
import com.tianyichen.cRBM.cRBM;
import com.tianyichen.dao.DataStream;
import com.tianyichen.utils.otherTools;

public class UIcRBM {
	
	public static void main(String[] args){
		
		//String inputX="GSE4226_4227_X.csv";
		//String inputY="GSE4226_4227_Y.csv";
		
		//String inputX="german_numerX";
		//String inputY="german_numerY";
		String inputX="ijcnn1_X.out";
		String inputY="ijcnn1_Y.out";
		Map<String, RealMatrix> infoMap=DataStream.loadData(inputX, inputY);
		
		RealMatrix X=infoMap.get("X");
		RealMatrix Y=infoMap.get("Y");
		
		int num_features=X.getColumnDimension();
		int num_samples=X.getRowDimension();
		num_samples=Y.getRowDimension();
		
		System.out.println("input dataset size:");
		System.out.println("number of features: "+num_features+", number of samples:"+num_samples);

		cRBM crbm=new cRBM(num_features, num_features/2);
		
		int K=50;
		double alpha=0.1;
		crbm.setK(K);
		crbm.setAlpha(alpha);
		int training_epoches=100;
		int batch_size=21;
		
		crbm.rbmtrain(X, batch_size, training_epoches);
		
	}

}
