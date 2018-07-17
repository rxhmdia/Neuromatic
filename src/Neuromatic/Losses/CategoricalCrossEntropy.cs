using TensorFlow;

namespace Neuromatic.Losses
{
    /// <summary>
    /// <para>
    /// The categorical cross entropy function is used in neural networks that perform multiclass
    /// classification jobs. You typically use this function in combination with a softmax output layer.
    /// </para>
    /// <para>In fact, we assume you do. So make sure you use a softmax layer when using this loss function.</para>
    /// </summary>
    public class CategoricalCrossEntropy: LossFunction
    {
        /// <summary>
        /// Compiles the categorical cross entropy function
        /// </summary>
        /// <param name="graph">Graph to use for compilation</param>
        /// <param name="predictions">Output of the neural network</param>
        /// <param name="targets">Targets to optimize towards</param>
        /// <returns>Returns the compiled loss function</returns>
        public override TFOutput Compile(TFGraph graph, TFOutput predictions, TFOutput targets)
        {
            var (loss,backprop) = graph.SoftmaxCrossEntropyWithLogits(predictions, targets);

            return loss;
        }
    }
}