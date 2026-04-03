class OrGateLayer(nn.Module):
         if not (0.0 <= threshold <= 1.0):
             raise ValueError(f"threshold must be in [0, 1], got {threshold}")
         with torch.no_grad():
-            discrete_w = torch.where(self.weight<threshold,torch.clamp_min_(self.weight,0),torch.clamp_max_(self.weight,1))#(self.weight >= threshold).float()
+            # Evaluation of clamp_max_ and clamp_min_ in-place would modify the whole tensor unpredictably.
+            # Using out-of-place equivalents:
+            # For weights < threshold, ensure they are <= 0 (clamp_max(..., 0))
+            # For weights >= threshold, ensure they are >= 1 (clamp_min(..., 1))
+            discrete_w = torch.where(
+                self.weight < threshold,
+                torch.clamp_max(self.weight, 0),
+                torch.clamp_min(self.weight, 1)
+            )

I had logical error on when i called discretize,But it still worked in tha valid way ,I think,I need to check if that's what it is?
