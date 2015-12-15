import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.linear.RealMatrix;

import com.tianyichen.RBM.RBM;
import com.tianyichen.dao.DataStream;
import com.tianyichen.utils.otherTools;


public class test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String inputX="code/german_numer01_X.out";
		String inputY="code/german_numer01_Y.out";
		
		Map<String, RealMatrix> infoMap=DataStream.loadData(inputX, inputY);
		
		RealMatrix X=infoMap.get("X");
		RealMatrix Y=infoMap.get("Y");
		int num_features=X.getColumnDimension();
		int num_samples=X.getRowDimension();
		int hidden_size=num_features/2;

		RBM rbm=new RBM(num_features, hidden_size);

		int K=1;
		double alpha=0.1;
		rbm.setK(K);
		rbm.setAlpha(alpha);
		int training_epoches=100;
		int batch_size=111;
		
		// train RBM
		for(int t=0;t<training_epoches;t++){
			Random randomGenerator = new Random();
			
			// select a batch of data randomly
			RealMatrix inputData=otherTools.stochasticSubmatrix(X, batch_size, randomGenerator);
			
			// update parameters of rbm1 with SGD
			rbm.updateParams(inputData);
			//rbm1.getVbias();
			double loss=rbm.CrossEntropy(inputData);
			System.out.println("training epoch:"+t+" reconstruction loss value:"+loss);
			//System.out.println("training epoch:"+t+" loss value:"+loss);

			
		}
		System.out.println("RBM size visible unit size:"+num_features+" hidden unit size:"+hidden_size);
		System.out.println("Contrasive divergence K:"+K);
	}

}
