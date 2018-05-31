using Neuromatic.Core;
using Neuromatic.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.Core
{
    /// <summary>
    /// Visitor used to compile the model to an executable graph
    /// </summary>
    public abstract class ModelBackend: IDisposable
    {
        /// <summary>
        /// Visit a model
        /// </summary>
        /// <param name="model">Model to visit</param>
        public abstract void CreateModel(Model model);

        /// <summary>
        /// Visits an input layer
        /// </summary>
        /// <param name="layer">Input layer to visit</param>
        public abstract void Input(Input layer);

        /// <summary>
        /// Gets the output for the visitor
        /// </summary>
        public abstract ExecutableModel Output { get; }

        #region IDisposable Support
        
        protected virtual void Dispose(bool disposing)
        {
            
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            Dispose(true);
        }
        #endregion
    }
}
