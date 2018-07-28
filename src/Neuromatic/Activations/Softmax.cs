using TensorFlow;

namespace Neuromatic.Activations
{
    /// <summary>
    /// The softmax function takes all input values and squashes them so that the total of all inputs sums up to 1.
    /// The output of the softmax function is a categorical probability distribution. When used as an activation on
    /// the output layer of your model, the element in the output vector with the highest value is the most likely
    /// category the input of your network is classified to be.
    /// </summary>
    public class Softmax: ActivationFunction
    {
        /// <summary>
        /// Compiles the activation function
        /// </summary>
        /// <param name="context">Use this context to register trainable parameters
        /// and build the computational graph for the layer</param>
        /// <param name="input">Input for the activation function</param>
        /// <returns>Returns the compiled activation function</returns>
        public override TFOutput Compile(ModelCompilationContext context, TFOutput input)
        {
            return context.Graph.Softmax(input);
        }
    }
}