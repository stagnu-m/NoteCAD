using System;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Assets.Code.Solver;
using Assets.Code.Tools;

public class EquationSystem  {

	public enum SolveResult {
		DEFAULT,
		OKAY,
		DIDNT_CONVEGE,
		REDUNDANT,
		POSTPONE,
		BREAK,
		ITERATION
	}

	bool isDirty = true;

	public bool IsDirty { get { return isDirty; } }
	public int maxSteps = 20;
	public int dragSteps = 3;
	public bool revertWhenNotConverged = true;
	public bool IsL1Norm => ChooseNormComponent.IsOldNorm();

	Exp[,] J;
	double[,] A;
	double[,] AAT;
	double[] B;
	double[] X;
	double[] Z;
	double[] oldParamValues;

	List<Exp> sourceEquations = new List<Exp>();
	List<Param> parameters = new List<Param>();

	List<Exp> equations = new List<Exp>();
	List<Param> currentParams = new List<Param>();

	Dictionary<Param, Param> subs;

	public void AddEquation(Exp eq) {
		sourceEquations.Add(eq);
		isDirty = true;
	}

	public void AddEquation(ExpVector v) {
		sourceEquations.Add(v.x);
		sourceEquations.Add(v.y);
		sourceEquations.Add(v.z);
		isDirty = true;
	}

	public void AddEquations(IEnumerable<Exp> eq) {
		sourceEquations.AddRange(eq);
		isDirty = true;
	}

	public void RemoveEquation(Exp eq) {
		sourceEquations.Remove(eq);
		isDirty = true;
	}

	public void AddParameter(Param p) {
		parameters.Add(p);
		isDirty = true;
	}

	public void AddParameters(IEnumerable<Param> p) {
		parameters.AddRange(p);
		isDirty = true;
	}

	public void RemoveParameter(Param p) {
		parameters.Remove(p);
		isDirty = true;
	}

	public void Eval(ref double[] B_, ref List<Exp> equations_, bool clearDrag) {
		for(int i = 0; i < equations_.Count; i++) {
			if(clearDrag && equations_[i].IsDrag()) {
				B_[i] = 0.0;
				continue;
			}
			B_[i] = equations_[i].Eval();
		}
	}

	public bool IsConverged(ref double[] B_, bool checkDrag, ref List<Exp> equations_, bool printNonConverged = false) {
		for(int i = 0; i < equations_.Count; i++) {
			if(!checkDrag && equations_[i].IsDrag()) {
				continue;
			}
			if(Math.Abs(B_[i]) < GaussianMethod.epsilon) continue;	
			if(printNonConverged) {
				Debug.Log("Not converged: " + equations_[i].ToString());
			}
			return false;
		}
		return true;
	}

	//void StoreParams() {
	//	for(int i = 0; i < parameters.Count; i++) {
	//		oldParamValues[i] = parameters[i].value;
	//	}
	//}

	void StoreParams(ref double[] oldParamValues_, List<Param> parameters_) {
		for(int i = 0; i < parameters_.Count; i++) {
			oldParamValues_[i] = parameters_[i].value;
		}
	}

	//void RevertParams() {
	//	for(int i = 0; i < parameters.Count; i++) {
	//		parameters[i].value = oldParamValues[i];
	//	}
	//}

	void RevertParams(ref double[] oldParamValues_, List<Param> parameters_) {
		for(int i = 0; i < parameters_.Count; i++) {
			parameters_[i].value = oldParamValues_[i];
		}
	}

	static Exp[,] WriteJacobian(ref List<Exp> equations_, ref List<Param> parameters_) {
		var J = new Exp[equations_.Count, parameters_.Count];
		for(int r = 0; r < equations_.Count; r++) {
			var eq = equations_[r];
			for(int c = 0; c < parameters_.Count; c++) {
				var u = parameters_[c];
				J[r, c] = eq.Deriv(u);
				/*
				if(!J[r, c].IsZeroConst()) {
					Debug.Log(J[r, c].ToString() + "\n");
				}
				*/
			}
		}
		return J;
	}

	public bool HasDragged() {
		return equations.Any(e => e.IsDrag());
	}

	public void EvalJacobian(
		ref List<Exp> equations_,
		ref List<Param> currentParams_,
		ref Dictionary<Param, Param> subs_,
		ref Exp[,] J_,
		ref double[,] A_,
		ref double[] B_,
		ref double[] X_,
		ref double[] Z_,
		ref double[,] AAT_,
		ref double[] oldParamValues_,
		bool clearDrag) {
		UpdateDirty(
			ref equations_,
			ref currentParams_,
			ref subs_,
			ref J_,
			ref A_,
			ref B_,
			ref X_,
			ref Z_,
			ref AAT_,
			ref oldParamValues_);	
		UnityEngine.Profiling.Profiler.BeginSample("EvalJacobian");
		for(int r = 0; r < J_.GetLength(0); r++) {
			if(clearDrag && equations_[r].IsDrag()) {
				for(int c = 0; c < J_.GetLength(1); c++) {
					A_[r, c] = 0.0;
				}
				continue;
			}
			for(int c = 0; c < J_.GetLength(1); c++) {
				A_[r, c] = J_[r, c].Eval();
			}
		}
		UnityEngine.Profiling.Profiler.EndSample();
	}

	public void SolveLeastSquares(double[,] A, double[] B, ref double[] X) {

		// A^T * A * X = A^T * B
		var rows = A.GetLength(0);
		var cols = A.GetLength(1);

		UnityEngine.Profiling.Profiler.BeginSample("SolveLeastSquares: A^T * A");
		var time = Time.realtimeSinceStartup;
		for(int r = 0; r < rows; r++) {
			for(int c = 0; c < rows; c++) {
				double sum = 0.0;
				for(int i = 0; i < cols; i++) {
					if(A[c, i] == 0 || A[r, i] == 0) continue;
					sum += A[r, i] * A[c, i];
				}
				AAT[r, c] = sum;
			}
		}
		//Debug.Log("AAT time " + (Time.realtimeSinceStartup - time) * 1000);
		UnityEngine.Profiling.Profiler.EndSample();

		GaussianMethod.Solve(AAT, B, ref Z);

		for(int c = 0; c < cols; c++) {
			double sum = 0.0;
			for(int r = 0; r < rows; r++) {
				sum += Z[r] * A[r, c];
			}
			X[c] = sum;
		}

	}

	public void Clear() {
		parameters.Clear();
		currentParams.Clear();
		equations.Clear();
		sourceEquations.Clear();
		isDirty = true;
		UpdateDirty(
			ref equations,
			ref currentParams,
			ref subs,
			ref J,
			ref A,
			ref B,
			ref X,
			ref Z,
			ref AAT,
			ref oldParamValues);
	}

	public bool TestRank(out int dof) {
		EvalJacobian(
			ref equations,
			ref currentParams,
			ref subs,
			ref J,
			ref A,
			ref B,
			ref X,
			ref Z,
			ref AAT,
			ref oldParamValues,
			clearDrag: false);
		int rank = GaussianMethod.Rank(A);
		dof = A.GetLength(1) - rank;
		return rank == A.GetLength(0);
	}

	public int GetDoF(double[,] matrix)
	{
		int rank = GaussianMethod.Rank(matrix);
		int dof = matrix.GetLength(1) - rank;
		return dof;
	}

	void UpdateDirty(ref List<Exp> equations_,
	                 ref List<Param> currentParams_,
	                 ref Dictionary<Param, Param> subs_,
	                 ref Exp[,] J_,
	                 ref double[,] A_,
	                 ref double[] B_,
	                 ref double[] X_,
	                 ref double[] Z_,
	                 ref double[,] AAT_,
	                 ref double[] oldParamValues_
		) {
		if(isDirty) {
			equations_ = sourceEquations.Select(e => e.DeepClone()).ToList();
			currentParams_ = parameters.ToList();
			/*
			foreach(var e in equations) {
				e.ReduceParams(currentParams);
			}*/
			//currentParams = parameters.Where(p => equations.Any(e => e.IsDependOn(p))).ToList();
			subs_ = SolveBySubstitution(ref equations_, ref currentParams_);

			J_ = WriteJacobian(ref equations_, ref currentParams_);
			A_ = new double[J_.GetLength(0), J_.GetLength(1)];
			B_ = new double[equations_.Count];
			X_ = new double[currentParams_.Count];
			Z_ = new double[A_.GetLength(0)];
			AAT_ = new double[A_.GetLength(0), A_.GetLength(0)];
			oldParamValues_ = new double[parameters.Count];
			isDirty = false;
			dofChanged = true;
		}
	}

	void BackSubstitution(ref Dictionary<Param, Param> subs_, ref List<Param> parameters_) {
		if(subs_ == null) return;
		for(int i = 0; i < parameters_.Count; i++) {
			var p = parameters_[i];
			if(!subs_.ContainsKey(p)) continue;
			p.value = subs_[p].value;
		}
	}

	Dictionary<Param, Param> SolveBySubstitution(
		ref List<Exp> equations_,
		ref List<Param> currentParams_) {
		var subs_ = new Dictionary<Param, Param>();

		for(int i = 0; i < equations_.Count; i++) {
			var eq = equations_[i];
			if(!eq.IsSubstitionForm()) continue;
			var a = eq.GetSubstitutionParamA();
			var b = eq.GetSubstitutionParamB();
			if(Math.Abs(a.value - b.value) > GaussianMethod.epsilon) continue;
			if(!currentParams_.Contains(b)) {
				var t = a;
				a = b;
				b = t;
			}
			// TODO: Check errors
			//if(!parameters.Contains(b)) {
			//	continue;
			//}

			foreach(var k in subs_.Keys.ToList()) {
				if(subs_[k] == b) {
					subs_[k] = a;
				}
			}
			subs_[b] = a;
			equations_.RemoveAt(i--);
			currentParams_.Remove(b);

			for(int j = 0; j < equations_.Count; j++) {
				equations_[j].Substitute(b, a);
			}
		}
		return subs_;
	}

	Dictionary<Param, Param> SolveBySubstitutionWithLocal(
		ref List<Exp> equations,
		ref List<Param> currentParams) {
		var subs = new Dictionary<Param, Param>();

		for(int i = 0; i < equations.Count; i++) {
			var eq = equations[i];
			if(!eq.IsSubstitionForm()) continue;
			var a = eq.GetSubstitutionParamA();
			var b = eq.GetSubstitutionParamB();
			if(Math.Abs(a.value - b.value) > GaussianMethod.epsilon) continue;
			if(!currentParams.Contains(b)) {
				var t = a;
				a = b;
				b = t;
			}
			// TODO: Check errors
			//if(!parameters.Contains(b)) {
			//	continue;
			//}

			foreach(var k in subs.Keys.ToList()) {
				if(subs[k] == b) {
					subs[k] = a;
				}
			}
			subs[b] = a;
			equations.RemoveAt(i--);
			currentParams.Remove(b);

			for(int j = 0; j < equations.Count; j++) {
				equations[j].Substitute(b, a);
			}
		}
		return subs;
	}

	public string stats { get; private set; }
	public bool dofChanged { get; private set; }

	public SolveResult Solve() {
		


		dofChanged = false;
		UpdateDirty(
			ref equations,
			ref currentParams,
			ref subs,
			ref J,
			ref A,
			ref B,
			ref X,
			ref Z,
			ref AAT,
			ref oldParamValues);
		StoreParams(ref oldParamValues, parameters);

		#region initialVars
		var equations_copy = equations.ToList();
		var currentParams_copy = new List<Param>();
		foreach (var currentParam in currentParams)
		{
			var currentParamCopy = currentParam.DeepClone();
			currentParams_copy.Add(currentParamCopy);
		}
		var params_copy = new List<Param>();
		foreach (var currentParam in parameters)
		{
			var currentParamCopy = currentParam.DeepClone();
			currentParams_copy.Add(currentParamCopy);
		}
		Exp[,] J_copy = J?.Clone() as Exp[,];
		double[,] A_copy = A?.Clone() as double[,];
		double[,] AAT_copy = AAT?.Clone() as double[,];
		double[] B_copy = B?.Clone() as double[];
		double[] X_copy = X?.Clone() as double[];
		double[] Z_copy = Z?.Clone() as double[];
		double[] oldParamValues_copy = oldParamValues?.Clone() as double[];
		var subs_copy = DeepCopyDictionary(subs);
		#endregion

		int steps = 0;

		//var dof = 0;
		bool isInitial = true;
		var initialSolution = new double[X.Length];
		var initialParamPosition = new double[X.Length];
		List<IndexToBlock> indicesToConsider = new List<IndexToBlock>();

		var dragIndices = new List<int>();

		for (int i = 0; i < currentParams.Count; i++)
		{
			initialParamPosition[i] = currentParams[i].value;
			if (!currentParams[i].IsDrag) continue;
			var currentParamIndex = currentParams.FindIndex(x => x == currentParams[i]);
			dragIndices.Add(currentParamIndex);
		}

		do
		{
			//IndexToBlock indexToBlock = null;
			var indexToBlock = indicesToConsider.FirstOrDefault(x => !x.IsUsed);
			//if (!isInitial)
			//{
			//var indexToBlock = new IndexToBlock { Index_x = 4, Index_y = 5 };
			//}

			steps = 0;
			var completeSolution = new double[X.Length];
			do
			{
                var result = SolveResult.DEFAULT;
                if (isInitial)
                {
                    result = SolveResultTest(
                        ref steps,
                        ref initialSolution,
                        ref initialParamPosition,
                        ref indicesToConsider,
                        ref dragIndices,
                        ref indexToBlock,
                        ref completeSolution,
                        ref isInitial,
                        ref equations,
                        ref subs,
                        ref parameters,
                        ref currentParams,
                        ref J,
                        ref A,
						ref B,
	                    ref X,
                        ref Z,
                        ref AAT,
                        ref oldParamValues);
                }

                if (result == SolveResult.BREAK)
                {
	                break;
                }

                if (!isInitial)
                {
	                result = SolveResultTest(
		                ref steps,
		                ref initialSolution,
		                ref initialParamPosition,
		                ref indicesToConsider,
		                ref dragIndices,
		                ref indexToBlock,
		                ref completeSolution,
		                ref isInitial,
		                ref equations_copy,
		                ref subs_copy,
		                ref params_copy,
		                ref currentParams_copy,
		                ref J_copy,
		                ref A_copy,
		                ref B_copy,
		                ref X_copy,
		                ref Z_copy,
		                ref AAT_copy, 
		                ref oldParamValues_copy);
                }

                if (result == SolveResult.ITERATION) continue;
                if (result == SolveResult.OKAY)
                {
	                //if (!isInitial)
	                //{
	                // for (int i = 0; i < currentParams.Count; i++)
	                // {
	                //  currentParams[i].value = currentParams_copy[i].value;
	                // }
	                //}
	                return result;
                }
            } while(steps++ <= maxSteps);
		} while (indicesToConsider.Count(x => !x.IsUsed) > 0);

		//var temp2 = currentParams.Count > 4 && Math.Abs(currentParams[4].value + 1.0) > Double.Epsilon;
		//if (temp2)
		//{
		//	var test = 0.0;
		//}

		//if (!isInitial)
		//{
		//	for (int i = 0; i < currentParams.Count; i++)
		//	{
		//		currentParams[i].value = initialSolution[i];
		//	}
		//}


		//if (IsConverged(checkDrag: false, printNonConverged: true))
		//{
		//	if (steps > 0)
		//	{
		//		dofChanged = true;
		//		Debug.Log(
		//			$"solved {equations.Count} equations with {currentParams.Count} unknowns in {steps} steps");
		//	}
		//	stats = $"eqs:{equations.Count}\nunkn: {currentParams.Count}";
		//	BackSubstitution(subs);
		//	return SolveResult.OKAY;
		//}

		//var notConverge = IsConverged(checkDrag: false, printNonConverged: true);
		//Debug.Log($"DIDNT_CONVEGE is {notConverge}");
		IsConverged(ref B,checkDrag: false, ref equations, printNonConverged: true);
		if (revertWhenNotConverged) {
			RevertParams(ref oldParamValues, parameters);
			dofChanged = false;
		}
		return SolveResult.DIDNT_CONVEGE;
	}

	private SolveResult SolveResultTest(ref int steps,
	                                    ref double[] initialSolution,
	                                    ref double[] initialParamPosition,
	                                    ref List<IndexToBlock> indicesToConsider,
	                                    ref List<int> dragIndices,
	                                    ref IndexToBlock indexToBlock,
	                                    ref double[] completeSolution,
	                                    ref bool isInitial,
	                                    ref List<Exp> equations_,
	                                    ref Dictionary<Param, Param> subs_,
	                                    ref List<Param> params_,
	                                    ref List<Param> currentParams_,
	                                    ref Exp[,] J_,
	                                    ref double[,] A_,
	                                    ref double[] B_,
	                                    ref double[] X_,
	                                    ref double[] Z_,
	                                    ref double[,] AAT_,
	                                    ref double[] oldParamValues_)
	{
		int dof = 0;
		bool isDragStep = steps <= dragSteps;
		//if (!isDragStep && !isInitial)
		//{
		//	if (!IsConverged(checkDrag: true))
		//	{
		//		break;
		//	}
		//}
		Eval(ref B_, ref equations_, clearDrag: !isDragStep);


		// remove to solve over-constraint systems
		//if (steps > 0)
		//{
		//	BackSubstitution(subs);
		//	return SolveResult.POSTPONE;
		//}

		if (IsConverged(ref B_, checkDrag: isDragStep, ref equations_))
		{
			//if (isInitial && steps > 0)
			//{
			//	//BlockFreeCoordinates(completeSolution, dragIndices, dof, indicesToConsider);
			//	isInitial = false;
			//	//if (indicesToConsider.Count > 0)
			//	//{
			//		for (int i = 0; i < initialSolution.Length; i++)
			//		{
			//			initialSolution[i] = currentParams_[i].value;
			//			//currentParams_[i].value = initialParamPosition[i];
			//		}

			//	//isDirty = true;
			//	//RevertParams();
			//	//UpdateDirty();
			//	//StoreParams();
			//	//X_for_method = new double[currentParams_for_method.Count];
			//	indicesToConsider.Add(new IndexToBlock { Index_x = 4, Index_y = 5 });
			//	return SolveResult.BREAK;
			//	//}
			//}

			if (steps > 0)
			{
				dofChanged = true;
				Debug.Log(
					$"solved {equations_.Count} equations with {currentParams_.Count} unknowns in {steps} steps");
			}

			//var temp = currentParams.Count > 4 && Math.Abs(currentParams[4].value + 1.0) > Double.Epsilon;
			//if (temp)
			//{
			//	for (int i = 0; i < initialParamPosition.Length; i++)
			//	{
			//		currentParams[i].value = initialParamPosition[i];
			//	}
			//	var test = 0.0;
			//}
			stats = $"eqs:{equations_.Count}\nunkn: {currentParams_.Count}";
			BackSubstitution(ref subs_, ref params_);
			return SolveResult.OKAY;
		}

		EvalJacobian(
			ref equations_,
			ref currentParams_,
			ref subs_,
			ref J_,
			ref A_,
			ref B_,
			ref X_,
			ref Z_,
			ref AAT_,
			ref oldParamValues_,
			clearDrag: !isDragStep);

		// TODO rewrite to solve for l_1

		//Debug.Log($"Matrix A before solve:\n{MatrixToString(A)}");
		//Debug.Log($"Vector B before solve:\n{string.Join(", ", B)}");
		//Debug.Log($"Vector X before solve:\n{string.Join(", ", X)}");
		var copyA = DeepCopyMatrix(A_);
		dof = GetDoF(copyA);
		Debug.Log($"Degree of Freedom is: {dof} \n");
		//if (IsL1Norm)
		//{
		LinearSolverExample.SolveLinearProgramInitial(ref A_, ref B_, ref X_, dragIndices, dof, indexToBlock);
		//}
		//else
		//{
		//	SolveLeastSquares(A, B, ref X);
		//}

		//Debug.Log($"Matrix A after solve:\n{MatrixToString(A)}");
		//Debug.Log($"Vector B after solve:\n{string.Join(", ", B)}");
		//Debug.Log($"Vector X after solve:\n{string.Join(", ", X)}");

		for (int i = 0; i < currentParams_.Count; i++)
		{
			currentParams_[i].value -= X_[i];
			completeSolution[i] -= X_[i];
		}
		return SolveResult.ITERATION;
	}

	#region Helper functions

    private static void BlockFreeCoordinates(double[] X, List<int> dragIndices, int dof, List<IndexToBlock> indicesToConsider)
    {
	    // Number of points in X (each point has two components)
	    int numPairs = X.Length / 2;

	    // If no DOF, exit early
	    if (dof == 0) return;

	    if (indicesToConsider.Count >= dof) return;

	    // Iterate over all pairs (points) in X
	    for (int i = 0; i < numPairs; i++)
	    {
		    // Skip drag indices
		    int varIndex_x = 2 * i;
		    int varIndex_y = 2 * i + 1;
		    if (dragIndices.Contains(varIndex_x) || dragIndices.Contains(varIndex_y)) continue;

		    // Current point components
		    double x_coordinate = X[varIndex_x];
		    double y_coordinate = X[varIndex_y];

		    // Skip if both coordinates are zero or both are non-zero
		    if (x_coordinate == 0 ^ y_coordinate == 0)
		    {
			    //TODO
			    indicesToConsider.Add(new IndexToBlock
			    {
				    Index_x = varIndex_x,
				    Value_x = x_coordinate,
				    Index_y = varIndex_y,
				    Value_y = y_coordinate,
				    IsUsed = false
			    });
		    }
		    //indicesToConsider.Sort((a, b) => a.AbsoluteSum.CompareTo(b.AbsoluteSum));
	    }

		indicesToConsider = indicesToConsider.OrderBy(x => x.AbsoluteSum).ToList();

	}

    // Create a method to print the 2D array
    string MatrixToString(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int columns = matrix.GetLength(1);
        string result = "";

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                result += matrix[i, j].ToString("F2") + "\t"; // Format with 2 decimal points
            }
            result += "\n";
        }

        return result;
    }

    private static double[,] DeepCopyMatrix(double[,] original)
    {
        int rows = original.GetLength(0);
        int cols = original.GetLength(1);
        double[,] copy = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                copy[i, j] = original[i, j];
            }
        }
        return copy;
    }

    private static double[] DeepCopyArray(double[] original)
    {
        double[] copy = new double[original.Length];

        for (int i = 0; i < original.Length; i++)
        {
            copy[i] = original[i];
        }
        return copy;
    }

    Dictionary<Param, Param> DeepCopyDictionary(Dictionary<Param, Param> original)
    {
	    var deepCopy = new Dictionary<Param, Param>();

	    foreach (var kvp in original)
	    {
		    // Deep clone both key and value
		    var clonedKey = kvp.Key.DeepClone();
		    var clonedValue = kvp.Value.DeepClone();
		    deepCopy[clonedKey] = clonedValue;
	    }

	    return deepCopy;
    }


    #endregion

}
