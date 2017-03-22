package Experiment;

import org.apache.commons.math3.linear.*;

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
    // Get Class Num
    int getClass_Num();
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

    public RealVector getTaskLabelProb(int task_no, boolean Prob_or_Not)
    {
        if(Prob_or_Not)
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