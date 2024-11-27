
package com.example.audiva

import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import android.widget.ImageView
import android.widget.TextView

class AnalysisAdapter(
    private val context: Context,
    private val analysisList: List<PhoneNumberAnalysis>
) : BaseAdapter() {

    private val inflater: LayoutInflater = LayoutInflater.from(context)

    override fun getCount(): Int = analysisList.size

    override fun getItem(position: Int): Any = analysisList[position]

    override fun getItemId(position: Int): Long = position.toLong()

    override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
        val view = convertView ?: inflater.inflate(R.layout.list_item_analysis, parent, false)

        // Get views
        val statusIcon = view.findViewById<ImageView>(R.id.statusIcon)
        val phoneNumber = view.findViewById<TextView>(R.id.phoneNumber)

        // Get data for this item
        val analysis = analysisList[position]

        // Set phone number text
        phoneNumber.text = analysis.phoneNumber

        // Set icon based on AI detection
        statusIcon.setImageResource(
            if (analysis.isAiVoiced) R.drawable.caution_7805444 else R.drawable.check_5610944
        )

        return view
    }
}
