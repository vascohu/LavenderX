package Experiment;

import org.apache.commons.math3.linear.*;
import org.jetbrains.annotations.Contract;
import java.util.Arrays;

/* The Probability Model memorizes the probability model of a crowd labeling system
 * Created by Zehong on 3/21/2017 0021.
 */
interface Prob_Model{
    // Get Label Probability calculates the label distribution caused by an action.
    RealVector getLabelProb(Action a);
    // Get Task Label Probability calculates the task label distribution.
    RealVector getTaskLabelProb(int task_no, boolean Prob_or_Not);
    // Update accurately calculates the probability model corresponding to a new state.
    void Update(Action a, State newS);
    // QuickUpdate roughly calculates the new probability model with high efficiency.
    void QuickUpdate(Action a, State newS);
    // Deep copy a new probability model
    Prob_Model Copy();
    // Get Task Num
    int getTask_Num();
    // Get Worker Num
    int getWorker_Num();
    // Get Class Num
    int getClass_Num();
}


abstract class Push_Market_Model implements Prob_Model
{
    final int m_worker_num;
    final int m_task_num;
    final int m_class_num;

    Push_Market_Model(int task_num, int worker_num, int class_num)
    {
        m_task_num = task_num;
        m_worker_num = worker_num;
        m_class_num = class_num;
    }

    Push_Market_Model(Push_Market_Model model)
    {
        m_task_num = model.m_task_num;
        m_worker_num = model.m_worker_num;
        m_class_num = model.m_class_num;
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
}


/* Majority Voting probability model
 */
class MJ_Model extends Push_Market_Model {

    // We assume the task labels follow the beta distribution
    private double[][] m_label_count, m_label_prob;

    MJ_Model(int task_num, int worker_num, final int class_num)
    {
        super(task_num, worker_num, class_num);
        m_label_count = new double[task_num][class_num];
        m_label_prob = new double[task_num][class_num];
        for(int i=0; i<task_num; ++i)
        {
            for(int j=0; j<class_num; ++j)
            {
                m_label_count[i][j] = 1;
                m_label_prob[i][j] = 1.0/class_num;
            }
        }
    }

    MJ_Model(MJ_Model model)
    {
        super(model.getTask_Num(), model.getWorker_Num(), model.getClass_Num());
        m_label_count = new double[m_task_num][m_class_num];
        m_label_prob = new double[m_task_num][m_class_num];
        for(int i=0; i<m_task_num; ++i)
        {
            System.arraycopy(model.m_label_count[i], 0, m_label_count[i], 0, m_class_num);
            System.arraycopy(model.m_label_prob[i], 0, m_label_prob[i], 0, m_class_num);
        }
    }

    public RealVector getLabelProb(Action a)
    {
        // double sum = Arrays.stream(m_label_count[a.i]).sum();
        // double[] prob = Arrays.stream(m_label_count[a.i]).map(s->s/sum).toArray();
        return new ArrayRealVector(m_label_prob[a.i]);
    }

    public RealVector getTaskLabelProb(int task_no, boolean Prob_or_Not)
    {
        if(Prob_or_Not)
        {
            // double sum = Arrays.stream(m_label_count[task_no]).sum();
            // double[] prob = Arrays.stream(m_label_count[task_no]).map(s->s/sum).toArray();
            return new ArrayRealVector(m_label_prob[task_no]);
        }
        else
        {
            return new ArrayRealVector(m_label_count[task_no]);
        }
    }

    public void Update(Action a, State newS)
    {
        int newLabel = (int)newS.getEntry(a.i, a.j);
        m_label_count[a.i][newLabel-1] += 1;
        double sum = Arrays.stream(m_label_count[a.i]).sum();
        for(int j=0;j<m_class_num;++j)
        {
            m_label_prob[a.i][j] = m_label_count[a.i][j]/sum;
        }
    }

    public void QuickUpdate(Action a, State newS)
    {
        Update(a, newS);
    }

    public MJ_Model Copy()
    {
        return new MJ_Model(this);
    }
}


/* Bayesian Update probability model -  JMLR 15
 */
class BU_Model extends Push_Market_Model {
    // The parameters for this probability model
    private double[] m_a,m_b,m_c,m_d;

    BU_Model(int task_num, int worker_num, int class_num)
    {
        super(task_num, worker_num, class_num);
        m_a = new double[task_num];
        m_b = new double[task_num];
        m_c = new double[worker_num];
        m_d = new double[worker_num];
        Arrays.fill(m_a, 1.0);
        Arrays.fill(m_b, 1.0);
        Arrays.fill(m_c, 4.0);
        Arrays.fill(m_d, 1.0);
    }

    BU_Model(BU_Model model)
    {
        super(model.getTask_Num(), model.getWorker_Num(), model.getClass_Num());
        m_a = new double[m_task_num];
        m_b = new double[m_task_num];
        m_c = new double[m_worker_num];
        m_d = new double[m_worker_num];
        System.arraycopy(model.m_a, 0, m_a, 0, m_task_num);
        System.arraycopy(model.m_b, 0, m_b, 0, m_task_num);
        System.arraycopy(model.m_c, 0, m_c, 0, m_worker_num);
        System.arraycopy(model.m_d, 0, m_d, 0, m_worker_num);
    }

    public BU_Model Copy()
    {
        return  new BU_Model(this);
    }

    public RealVector getLabelProb(Action a)
    {
        double[] prob = new double[2];
        double th = m_a[a.i]/(m_a[a.i]+m_b[a.i]);
        double rou = m_c[a.j]/(m_c[a.j]+m_d[a.j]);
        prob[0] = th*rou + (1-th)*(1-rou);
        prob[1] = 1 - prob[0];
        return new ArrayRealVector(prob);
    }

    public RealVector getTaskLabelProb(int task_no, boolean Prob_or_Para)
    {
        double[] prob = new double[2];
        prob[0] = m_a[task_no]/(m_a[task_no]+m_b[task_no]);
        prob[1] = 1 - prob[0];
        return new ArrayRealVector(prob);
    }

    public void Update(Action a, State newS)
    {
        double va = m_a[a.i];
        double vb = m_b[a.i];
        double vc = m_c[a.j];
        double vd = m_d[a.j];
        double Eth=0, EEth=0, Erou=0, EErou=0;
        if(newS.getEntry(a.i, a.j)==1)
        {
            Eth = (va*((va+1)*vc+vb*vd))/((va+vb+1)*(va*vc+vb*vd));
            EEth = (va*(va+1)*((va+2)*vc+vb*vd))/((va+vb+1)*(va+vb+2)*(va*vc+vb*vd));
            Erou = (vc*(va*(vc+1)+vb*vd))/((vc+vd+1)*(va*vc+vb*vd));
            EErou = (vc*(vc+1)*(va*(vc+2)+vb*vd))/((vc+vd+1)*(vc+vd+2)*(va*vc+vb*vd));
        }
        else if (newS.getEntry(a.i, a.j)==2)
        {
            Eth = (va*(vb*vc+(va+1)*vd))/((va+vb+1)*(vb*vc+va*vd));
            EEth = (va*(va+1)*(vb*vc+(va+2)*vd))/((va+vb+1)*(va+vb+2)*(vb*vc+va*vd));
            Erou = (vc*(vb*(vc+1)+va*vd))/((vc+vd+1)*(vb*vc+va*vd));
            EErou = (vc*(vc+1)*(vb*(vc+2)+va*vd))/((vc+vd+1)*(vc+vd+2)*(vb*vc+va*vd));
        }
        else
        {
            System.out.println("Get Wrong Action!!!");
            System.exit(1);
        }
        double ti_a = Eth*(Eth-EEth)/(EEth-Eth*Eth);
        double ti_b = (1-Eth)*(Eth-EEth)/(EEth-Eth*Eth);
        double ti_c = Erou*(Erou-EErou)/(EErou-Erou*Erou);
        double ti_d = (1-Erou)*(Erou-EErou)/(EErou-Erou*Erou);
        m_a[a.i] = ti_a;
        m_b[a.i] = ti_b;
        m_c[a.j] = ti_c;
        m_d[a.j] = ti_d;
    }

    public  void QuickUpdate(Action a, State newS)
    {
        Update(a, newS);
    }

    public void PrintModel()
    {
        for(int i=0; i<m_worker_num; ++i)
        {
            System.out.print(m_c[i]/(m_c[i]+m_d[i])+",\t");
        }
        System.out.println();
    }
}

/* EM probability model
 */
class EM_Model extends Push_Market_Model {

    private double[][][] c_tensor;
    private double[][] p_label;

    EM_Model(int task_num, int worker_num, int class_num)
    {
        super(task_num,worker_num,class_num);
        c_tensor = new double[worker_num][class_num][class_num];
        p_label = new double[task_num][class_num];
        InitModel();
    }

    private void InitModel()
    {
        for(double[][] p:c_tensor)
        {
            for(int k=0; k<m_class_num; k++)
            {
                for(int t=0; t<m_class_num; t++)
                {
                    if(k==t)
                    {
                        p[k][t] = 1.0;
                    }
                    else
                    {
                        p[k][t] = 0;
                    }
                }
            }
        }
        for(double[] p:p_label)
        {
            Arrays.fill(p, 0.5);
        }
    }

    EM_Model(EM_Model model)
    {
        super(model);
        c_tensor = new double[m_worker_num][m_class_num][m_class_num];
        p_label = new double[m_task_num][m_class_num];
        for(int i=0; i<m_worker_num; ++i)
        {
            for(int j=0; j<m_class_num; ++j)
            {
                System.arraycopy(model.c_tensor[i][j],0, c_tensor[i][j], 0, m_class_num);
            }
        }
        for(int i=0; i<m_task_num; ++i)
        {
            System.arraycopy(model.p_label[i], 0, p_label[i], 0, m_class_num);
        }
    }

    public RealVector getLabelProb(Action a)
    {
        double[] p = new double[m_class_num];
        for(int i=0; i<m_class_num; ++i)
        {
            p[i] = 0;
            for(int j=0; j<m_class_num; ++j)
            {
                p[i] += p_label[a.i][j]*c_tensor[a.j][j][i];
            }
        }
        return new ArrayRealVector(p);
    }

    public RealVector getTaskLabelProb(int task_no, boolean Prob_or_Para)
    {
        return new ArrayRealVector(p_label[task_no]);
    }

    private void PIteration(State S)
    {
        for(double[] p:p_label)
        {
            Arrays.fill(p,0.0);
        }
        for(int i=0; i<m_task_num; ++i)
        {
            for(int j=0; j<m_worker_num; ++j)
            {
                double ob_label = S.getEntry(i,j);
                if(ob_label>0.5)
                {
                    for(int k=0; k<m_class_num; ++k)
                    {
                        p_label[i][k] += Math.log(c_tensor[j][k][(int)ob_label-1]+1e-20);
                    }
                }
            }
            double prob_sum = 0;
            for(int k=0; k<m_class_num; ++k)
            {
                p_label[i][k] = Math.exp(p_label[i][k]);
                prob_sum += p_label[i][k];
            }
            for(int k=0; k<m_class_num; ++k)
            {
                p_label[i][k] /= prob_sum;
            }
        }
    }

    private void MIteration(State S)
    {
        for(double[][] p: c_tensor)
        {
            for(double[] pp:p)
            {
                Arrays.fill(pp, 0.0);
            }
        }
        for(int i=0; i<m_task_num; ++i)
        {
            for(int j=0; j<m_worker_num; ++j)
            {
                double ob_label = S.getEntry(i,j);
                if(ob_label>0.5)
                {
                    for(int k=0; k<m_class_num; ++k)
                    {
                        c_tensor[j][k][(int)ob_label-1] += p_label[i][k];
                    }
                }
            }
        }
        for(int j=0; j<m_worker_num; ++j)
        {
            for(int k=0; k<m_class_num; ++k)
            {
                double prob_sum = 0;
                for(int t=0; t<m_class_num; ++t)
                {
                    prob_sum += c_tensor[j][k][t];
                }
                if(prob_sum!=0)
                {
                    for(int t=0; t<m_class_num; ++t)
                    {
                        c_tensor[j][k][t] /= prob_sum;
                    }
                }
                else
                {
                    for(int t=0; t<m_class_num; ++t)
                    {
                        if(t==k)
                        {
                            c_tensor[j][k][t] = 1.0;
                        }
                        else
                        {
                            c_tensor[j][k][t] = 0.0;
                        }
                    }
                }
            }
        }
    }

    @Contract(pure = true)
    private double Calc_c_Diff(double[][][] c0)
    {
        double c_diff = 0;
        for(int j=0; j<m_worker_num; ++j)
        {
            for(int k=0; k<m_class_num; ++k)
            {
                for(int t=0; t<m_class_num; ++t)
                {
                    c_diff += Math.abs(c_tensor[j][k][t]-c0[j][k][t]);
                }
            }
        }
        return(c_diff);
    }

    private void c_clone(double[][][] cN)
    {
        for(int j=0; j<m_worker_num; ++j)
        {
            for(int k=0; k<m_class_num; ++k)
            {
                System.arraycopy(c_tensor[j][k], 0, cN[j][k], 0, m_class_num);
            }
        }
    }

    public void Update(Action a, State newS)
    {
        InitModel();
        double[][][] c0 = new double[m_worker_num][m_class_num][m_class_num];
        do {
            c_clone(c0);
            PIteration(newS);
            MIteration(newS);
        }while(Calc_c_Diff(c0)>1e-6);
        PIteration(newS);
    }

    public  void QuickUpdate(Action a, State newS)
    {
        for(int i=0; i<2; ++i)
        {
            PIteration(newS);
            MIteration(newS);
        }
        PIteration(newS);
    }

    public EM_Model Copy()
    {
        return new EM_Model(this);
    }

    void PrintModel()
    {
        for(int j=0; j<m_worker_num; ++j)
        {
            //System.out.print("Worker "+j+": ");
            double acc=0;
            for(int k=0; k<m_class_num; ++k)
            {
                acc += c_tensor[j][k][k];
            }
            System.out.print(acc/m_class_num+",\t");
        }
        System.out.println();
    }
}

/* Bayesian EM probability model
 */
class BEM_Model() extends Push_Market_Model {

}