package Experiment;

/* Active Mechanism is responsible for deciding the task-worker selection.
 * Created by Zehong on 3/16/2017 0016.
 */

import org.apache.commons.math3.linear.*;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;


public class Active_Mechanism {
    private Obj_Function m_obj;
    private RL_Decision m_RL;
    private Prob_Model m_PModel;
    private State m_St;
    private Market_Simulator m_simulator;
    List<Double> accuracy_record;

    Active_Mechanism(Market_Simulator simulator)
    {
        m_St = new State(simulator.getTask_Num(), simulator.getWorker_Num());
        for(int i=0; i<simulator.getTask_Num(); ++i)
        {
            for(int j=0; j<simulator.getWorker_Num(); ++j)
            {
                m_St.setEntry(i,j,0);
            }
        }
        m_simulator = simulator;
    }

    void Mechanism_SetUp(Obj_Function obj, RL_Decision rl_module, Prob_Model prob_model)
    {
        m_obj = obj;
        m_RL = rl_module;
        m_PModel = prob_model;
    }

    void Run(int T) throws InterruptedException {
        accuracy_record = new ArrayList<>();
        while (m_RL.getT() < T)
        {
            // Make the action decision
            Action a = m_RL.getDecision(m_St, m_PModel, m_obj);

            // Get the label from the market
            double label = m_simulator.getLabelStream(a.i, a.j);
            //System.out.println(m_RL.getT()+">>>\t(Task: "+a.i +", Worker: "+a.j+")\t--->\t"+label);

            // Update the state variable
            m_RL.UpdateSate(a, label);

            // Update the probability model
            m_PModel.Update(a, m_St);

            // Calculate the accuracy
            if(m_RL.getT()%50==0)
            {
                double accuracy = Calculate_Accuracy(getTaskLabel(), m_simulator.getTrue_label());
                accuracy_record.add(accuracy);
            }
        }
        m_RL.closeThreadPool();
    }


    public double getObjValue()
    {
        return m_obj.getObjValue(m_PModel);
    }

    public State getLabelMat()
    {
        return m_St;
    }

    public List<Integer> getTaskLabel()
    {
        List<Integer> task_label = new ArrayList<>(m_PModel.getClass_Num());
        for(int i=0; i<m_PModel.getTask_Num(); ++i)
        {
            RealVector prob_vec = m_PModel.getTaskLabelProb(i, true);
            task_label.add(prob_vec.getMaxIndex()+1);
        }
        return task_label;
    }

    private static double Calculate_Accuracy(List<Integer> learned_labels, RealVector true_labels)
    {
        int number_of_correct = 0;
        Iterator<Integer> it1 = learned_labels.iterator();
        double[] true_label_array = true_labels.toArray();
        int i = 0;
        while(it1.hasNext())
        {
            if(Math.abs(it1.next()-true_label_array[i])<0.5)
            {
                number_of_correct++;
            }
            ++i;
        }

        return (double)number_of_correct/learned_labels.size();
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

    public List<Action> getAvailableAction()
    {
        ArrayList<Action> available_action = new ArrayList<>();
        for(int i=0; i<this.getRowDimension(); ++i)
        {
            for(int j=0; j<this.getColumnDimension(); ++j)
            {
                if(this.getEntry(i,j)==0) {
                    available_action.add(new Action(i, j));
                }
            }
        }
        return available_action;
    }
}

/* Give an alias to the duplicated labeling table vector.
 */
class DVState {
    Vector<State> m_state_vec;

    DVState(State s, int d) {
        m_state_vec = new Vector<>();
        m_state_vec.add(s);
        for(int i=1; i<d; ++i)
        {
            m_state_vec.add(s.copy());
        }
    }

    void setEntry(int i, int j, double val)
    {
        for(State s: m_state_vec)
        {
            s.setEntry(i,j, val);
        }
    }

    void setEntry(int i, int j, int val)
    {
        for(State s: m_state_vec)
        {
            s.setEntry(i,j, (double)val);
        }
    }

    State get(int i)
    {
        return m_state_vec.get(i);
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