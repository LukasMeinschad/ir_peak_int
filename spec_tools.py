
# Module Import
from classes import spectrum



def plot_spectrum(self,title=None,interactive=True,color="black",legend_title="Legend"):
        """ 
        Plots the spectrum using Altair as a package
        
        Parameters:
            self: The Spectrum object
            title: The title of the plot
            interactive: If True, plot is interactive
            color: Gives the Color of the Data in the Plot
            legend_title: Title of the Legend in the Plot
        """
        
        alt.data_transformers.disable_max_rows()
        
        if not title:
            title = "Spectrum of " + self.name

        



        data = pd.DataFrame(self.data[:,0:2], columns=["x","y"])
        
        data["Legend"]  = self.name

        # Adjust axis scale

        min_x = data["x"].min() - 10
        max_x = data["x"].max() + 10

        chart = alt.Chart(data).mark_line().encode(
            x=alt.X("x", 
                    title="Wave Number / cm⁻¹", 
                    sort="descending",
                    axis = alt.Axis(format="0.0f"),
                    scale=alt.Scale(domain=[min_x,max_x])),
            
            y=alt.Y("y", title="Intensity"),
            
            color = alt.Color("legend:N",legend=alt.Legend(title="Spectrum Name"))
            #color=alt.value("black"),
        ).properties(
            title=title,
            width = 800,
            height = 400
        )

        # Create Selection

        if interactive==True:
            # Create Selection
            selection = alt.selection_interval(bind="scales")
            # Add selection to chart
            chart = chart.add_selection(selection)


        return chart