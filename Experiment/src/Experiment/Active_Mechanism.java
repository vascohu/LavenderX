package Experiment;

/* Active Mechanism is responsible for deciding the task-worker selection.
 * Created by Zehong on 3/16/2017 0016.
 */

import org.apache.commons.math3.linear.*;

import java.util.ArrayList;

import static java.lang.Math.log;

public class Active_Mechanism {
    private Obj_Function m_obj;
    private RL_Decision m_RL;
    private Prob_Model m_PModel;
    private State m_St;
    private Market_Simulator m_simulator;

    public Active_Mechanism(Market_Simulator simulator)
    {
        m_obj = new Confidence_Obj();

        ArrayList<Action> available_action = new ArrayList<>();
        m_St = new State(simulator.getTask_Num(), simulator.getWorker_Num());
        for(int i=0; i<simulator.getTask_Num(); ++i)
        {
            for(int j=0; j<simulator.getWorker_Num(); ++j)
            {
                available_action.add(new Action(i,j));
                m_St.setEntry(i,j,0);
            }
        }

        // m_RL = new Random_RL(0, 0, available_action);
        // m_RL = new OptGrad_RL(0, 0, available_action);
        m_RL = new EpsGrad_RL(0, 0, available_action);

        m_PModel = new MJ_Model(simulator);

        m_simulator = simulator;
    }

    public void Run(int T) throws InterruptedException {
        while (m_RL.getT() < T)
        {
            // Make the action decision
            Action a = m_RL.getDecision(m_St, m_PModel, m_obj);

            // Get the label from the market
            double label = m_simulator.getLabelStream(a.i, a.j);
            System.out.println(m_RL.getT()+">>>\t(Task: "+a.i +", Worker: "+a.j+")\t--->\t"+label);

            // Update the state variable
            m_St.setEntry(a.i, a.j, label);

            // Update the probability model
            m_PModel.Update(a, m_St);
        }
    }

    public double getFinalObjValue()
    {
        return m_obj.getObjValue(m_PModel);
    }

    public State getFinalLabelMat()
    {
        return m_St;
    }
}

/* Give an alias to the labeling table.
 */
class State extends BlockRealMatrix {
    public State(int row_num, int col_num)
    {
        super(row_num,col_num);
    }

    public State(State old_S)
    {
        super(old_S.getData());
    }

    public void print()
    {
        for(int i=0; i<this.getRowDimension(); ++i)
        {
            for(int j=0; j<this.getColumnDimension(); ++j)
            {
                System.out.print(this.getEntry(i,j)+" , ");
            }
            System.out.print("\n");
        }
    }

    public void Sparse_Print()
    {
        for(int i=0; i<this.getRowDimension(); ++i)
        {
            for(int j=0; j<this.getColumnDimension(); ++j)
            {
                if(this.getEntry(i,j)>0.5)
                {
                    System.out.println("(Task:"+i + ", Worker:"+j+")--->"+this.getEntry(i,j));
                }
            }
        }
    }

    public State copy()
    {
        return new State(this);
    }

}

/* Give an alias to the task-worker decision (i---task no, j---worker no).
 */
class Action {
    public int i,j;
    public Action(int selected_task_no, int selected_worker_no)
    {
        i = selected_task_no;
        j = selected_worker_no;
    }
}

/* The Objective Function calculates the desired accuracy metric.
 */
interface Obj_Function {
    double getObjValue(Prob_Model model);
}


/* The Probability Model memorizes the probability model of a crowd labeling system
 */
interface Prob_Model{
    // Get Label Probability calculates the label distribution caused by an action.
    RealVector getLabelProb(Action a);
    // Get Task Label Probability calculates the task label distribution.
    RealVector getTaskLabelProb(int task_no, boolean Prob_or_Para);
    // Update accurately calculates the probability model corresponding to a new state.
    void Update(Action a, State newS);
    // QuickUpdate roughly calculates the new probability model with high efficiency.
    void QuickUpdate(Action a, State newS);
    // Deep copy a new probability model
    Prob_Model Copy();
    // Get Task Num
    int getTask_Num();
    // Get Class Num
    int getClass_Num();
}


/* The confidence objective function calculates the confidence sum[Pr(l^{*}_{i})]
 */
class Confidence_Obj implements Obj_Function {
    @Override
    public double getObjValue(Prob_Model model) {
        int task_num = model.getTask_Num();
        double confidence = 0;
        for(int i=0; i<task_num; ++i)
        {
            RealVector ProbVec = model.getTaskLabelProb(i, true);
            confidence += ProbVec.getMaxValue();
        }
        return confidence;
    }
}

/* The margin objective function calculates the margin sum[Pr(l^{*}_{i})-Pr(l^{*-1}_{i})]
 */
class Margin_Obj implements Obj_Function {
    @Override
    public double getObjValue(Prob_Model model) {
        int task_num = model.getTask_Num();
        double margin = 0;
        for(int i=0; i<task_num; ++i)
        {
            RealVector ProbVec = model.getTaskLabelProb(i, true);
            double max_confidence = ProbVec.getMaxValue();
            int index_of_max_conf = ProbVec.getMaxIndex();
            ProbVec.setEntry(index_of_max_conf, 0);
            double second_max_confidence = ProbVec.getMaxValue();
            margin += max_confidence - second_max_confidence;
        }
        return margin;
    }
}


/* The entropy objective function calculates the posterior entropy sum[Pr(l_{ik})*log(Pr(l_{ik}))]
 */
class Entropy_Obj implements Obj_Function {
    @Override
    public double getObjValue(Prob_Model model) {
        int task_num = model.getTask_Num();
        double entropy = 0;
        for(int i=0; i<task_num; ++i)
        {
            RealVector ProbVec = model.getTaskLabelProb(i, true);
            for(int k=0; k<ProbVec.getDimension(); ++k)
            {
                entropy += ProbVec.getEntry(k)*log(ProbVec.getEntry(k))-0.5*log(0.5);
            }
        }
        return entropy;
    }
}


/* Majority Voting probability model
 */
class MJ_Model implements Prob_Model {

    // We assume the task labels follow the beta distribution
    private RealMatrix m_label_count;
    private RealMatrix m_label_prob;

    private MJ_Model() {}

    private void Set_Model(RealMatrix label_count, RealMatrix label_prob)
    {
        m_label_count = label_count;
        m_label_prob = label_prob;
    }

    MJ_Model(final Market_Simulator simulator)
    {
        m_label_prob = new BlockRealMatrix(simulator.getTask_Num(), simulator.getClass_Num());
        m_label_count = new BlockRealMatrix(simulator.getTask_Num(), simulator.getClass_Num());
        m_label_prob.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
            @Override
            public double visit(int row, int column, double value)
            {
                return 1.0/(simulator.getClass_Num());
            }
        });
        m_label_count.walkInOptimizedOrder(new DefaultRealMatrixChangingVisitor() {
            @Override
            public double visit(int row, int column, double value)
            {
                return 1.0;
            }
        });
    }

    public RealVector getLabelProb(Action a)
    {
        return m_label_prob.getRowVector(a.i);
    }

    public RealVector getTaskLabelProb(int task_no, boolean Prob_or_Para)
    {
        if(Prob_or_Para)
        {
            return m_label_prob.getRowVector(task_no);
        }
        else
        {
            return m_label_count.getRowVector(task_no);
        }
    }

    public void Update(Action a, State newS)
    {
        int newLabel = (int)newS.getEntry(a.i, a.j);
        double count = m_label_count.getEntry(a.i, newLabel-1);
        m_label_count.setEntry(a.i, newLabel-1, count+1);
        RealVector count_vector = m_label_count.getRowVector(a.i);
        m_label_prob.setRowVector(a.i, count_vector.mapMultiply(1.0/count_vector.getL1Norm()));
    }

    public void QuickUpdate(Action a, State newS)
    {
        Update(a, newS);
    }

    public MJ_Model Copy()
    {
        MJ_Model new_Model = new MJ_Model();
        RealMatrix new_label_count = new BlockRealMatrix(m_label_count.getData());
        RealMatrix new_label_prob = new BlockRealMatrix(m_label_prob.getData());
        new_Model.Set_Model(new_label_count, new_label_prob);
        return new_Model;
    }

    public int getTask_Num()
    {
        return m_label_count.getRowDimension();
    }

    public int getClass_Num()
    {
        return m_label_count.getColumnDimension();
    }
}



/* EM probability model
 */
class EM_Model implements Prob_Model {

    public RealVector getLabelProb(Action a)
    {
        return new ArrayRealVector(3);
    }

    public RealVector getTaskLabelProb(int task_no, boolean Prob_or_Para)
    {
        return new ArrayRealVector(3);
    }

    public void Update(Action a, State newS)
    {

    }

    public  void QuickUpdate(Action a, State newS)
    {

    }

    private EM_Model() {}

    private void Set_Model() {}

    public EM_Model Copy() {return  new EM_Model();}

    public int getTask_Num()
    {
        return 0;
    }

    public int getClass_Num()
    {
        return 0;
    }
}