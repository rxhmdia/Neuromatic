using TensorFlow;

namespace Neuromatic.Activations
{
    /// <summary>
    /// <para>
    /// The ReLU activation is a pass-through activation function that clips any input that is below zero to zero.
    /// This activation function is widely used in almost all deep learning architectures.
    /// </para>
    /// <para>
    /// This activation is not always useful, especially if you have negative inputs that you need to preserve.
    /// The ReLU function immediately maps these input values to zero. It looses information in these instances.
    /// </para>
    /// </summary>
    public class RELU : ActivationFunction
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
            return context.Graph.Relu(input);
        }
    }
}