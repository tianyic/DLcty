import java.util.Map;

import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.DBN.DBN;
import com.tianyichen.RBM.RBM;
import com.tianyichen.dao.DataStream;


public class testDBN {
	
	public static void main(String[] args){
		String inputX="code/german_numer01_X.out";
		String inputY="code/german_numer01_Y.out";
		
		Map<String, RealMatrix> infoMap=DataStream.loadData(inputX, inputY);
		
		RealMatrix X=infoMap.get("X");
		RealMatrix Y=infoMap.get("Y");
		int num_features=X.getColumnDimension();
		int num_samples=X.getRowDimension();
		
		int batch_size=111;
		int K=1;
		int pre_training_epoches=10;
		int[] hiddensizes={10,100,50};
		//int[] hiddensizes={num_features/2};
		double alpha=0.1;
		DBN dbn=new DBN(num_features,hiddensizes,2,K,pre_training_epoches,alpha);
		// pretraining RBMs.
		dbn.pretraining(X, batch_size);

		System.out.print("DBN size:"+num_features+" ");
		for (int i=0;i<hiddensizes.length;i++ ) {
			System.out.print(hiddensizes[i]+" ");
		}
		//System.out.print(outputsize);
		System.out.println();
		//RBM rbm=dbn.getRbmlayers().get(0);
		//RBM rbm1=dbn.getRbmlayers().get(1);
	}
}
