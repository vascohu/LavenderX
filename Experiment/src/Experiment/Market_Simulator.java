package Experiment;

/* The Market_Simulation module is responsible for mimic the labeling process in real markets
 * Created by Zehong on 3/14/2017 0014.
 */

import jsc.distributions.*;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;


public class Market_Simulator {
    private int m_task_num, m_worker_num, m_class_num;
    private Task[] m_task_array;
    private Worker[] m_worker_array;
    private RealMatrix m_label_mat;
    private RealMatrix true_label_mat, reliability_mat;

    public Market_Simulator(int task_num, int worker_num)
    {
        m_task_num = task_num;
        m_worker_num = worker_num;
        m_class_num = 2;
        Generate_Market();
        Get_Ground_Truth();
    }

    public Market_Simulator(int task_num, int worker_num, int class_num)
    {
        m_task_num = task_num;
        m_worker_num = worker_num;
        m_class_num = class_num;
        Generate_Market();
        Get_Ground_Truth();
    }


    // Get the online label from the label mat
    public double getLabelStream(int task_no, int worker_no)
    {
        return m_label_mat.getEntry(task_no, worker_no);
    }

    public int getTask_Num()
    {
        return m_task_num;
    }

    public int getWorker_Num()
    {
        return m_worker_num;
    }

    public int getClass_Num()
    {
        return m_class_num;
    }

    public RealVector getTrue_label()
    {
        return true_label_mat.getColumnVector(0);
    }

    public RealMatrix getReliability_mat()
    {
        return reliability_mat;
    }

    // Generate workers and tasks for the market
    private void Generate_Market()
    {
        m_task_array = new Homo_Task[m_task_num];
        m_worker_array = new SingleCoin_Worker[m_worker_num];
        m_label_mat = new BlockRealMatrix(m_task_num, m_worker_num);
        for(int i=0; i<m_task_num; ++i)
        {
            m_task_array[i] = new Homo_Task();
        }
        for(int i=0; i<m_worker_num; ++i)
        {
            m_worker_array[i] = new SingleCoin_Worker();
        }
        for(int i=0; i<m_task_num; ++i)
        {
            for(int j=0; j<m_worker_num; ++j)
            {
                int label = m_worker_array[j].labelTask(m_task_array[i]);
                m_label_mat.setEntry(i,j,(double)label);
            }
        }
    }

    // Get the true labels and reliability data for evaluation
    private void Get_Ground_Truth()
    {
        true_label_mat = new BlockRealMatrix(m_task_num, 1);
        reliability_mat = new BlockRealMatrix(1, m_worker_num);
        for(int i=0; i<m_task_num; ++i)
        {
            int true_label = m_task_array[i].getTrueLabel();
            true_label_mat.setEntry(i,0, (double)true_label);
        }
        for(int i=0; i<m_worker_num; ++i)
        {
            String class_name = m_worker_array[i].getClass().getName();
            String single_coin_class_name = SingleCoin_Worker.class.getName();
            if(class_name.equals(single_coin_class_name))
            {
                double reliability = ((SingleCoin_Worker) m_worker_array[i]).getReliability();
                reliability_mat.setEntry(0, i, reliability);
            }
            else
            {
                reliability_mat.setEntry(0, i, -1);
            }
        }
    }
}

interface Worker {
    int labelTask(Task t);
}

interface Task {
    // getLabel() provides the label that can be observed by workers
    int getLabel();

    // getTrueLabel() provides the true label of tasks
    int getTrueLabel();
}

// Single Coin Worker generates correct labels with probability m_reliability
class SingleCoin_Worker implements Worker {

    private double m_reliability;

    SingleCoin_Worker()
    {
        double ub_reliability = 1.0;
        double lb_reliability = 0.4;
        Uniform random_ob = new Uniform(lb_reliability, ub_reliability);
        m_reliability = random_ob.random();
        //m_reliability = 0.8;
    }

    public int labelTask(Task t)
    {
        int ob_label = t.getLabel();
        int re_label;
        Bernoulli random_ob = new Bernoulli(m_reliability);
        double true_or_false = random_ob.random();
        if(true_or_false>0.5)
        {
            re_label = ob_label;
        }
        else
        {
            re_label = 3 - ob_label;
        }
        return re_label;
    }

    double getReliability()
    {
        return m_reliability;
    }
}

// Homogeneous Tasks have the same level of difficulty
class Homo_Task implements Task {

    private int m_label;

    Homo_Task()
    {
        Bernoulli random_ob = new Bernoulli(0.5);
        double true_or_false = random_ob.random();
        if(true_or_false >= 0.5)
        {
            m_label = 2;
        }
        else
        {
            m_label = 1;
        }
    }

    public int getLabel()
    {
        return m_label;
    }

    public int getTrueLabel()
    {
        return m_label;
    }
}
